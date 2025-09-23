#include <Kokkos_Core.hpp>
#include <type_traits>
#include <cstdint>
#include <unistd.h>
#include <iostream>

#ifdef LAPIS_USE_KOKKOSKERNELS
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosBlas3_gemm.hpp"
#include "KokkosBlas2_gemv.hpp"
#endif

template <typename T, int N>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

namespace LAPIS
{
  using TeamPolicy = Kokkos::TeamPolicy<>;
  using TeamMember = typename TeamPolicy::member_type;

  template<typename V>
  StridedMemRefType<typename V::value_type, V::rank> viewToStridedMemref(const V& v)
  {
    StridedMemRefType<typename V::value_type, V::rank> smr;
    smr.basePtr = v.data();
    smr.data = v.data();
    smr.offset = 0;
    for(int i = 0; i < int(V::rank); i++)
    {
      smr.sizes[i] = v.extent(i);
      smr.strides[i] = v.stride(i);
    }
    return smr;
  }

  template<typename V>
  V stridedMemrefToView(const StridedMemRefType<typename V::value_type, V::rank>& smr)
  {
    using Layout = typename V::array_layout;
    static_assert(std::is_same_v<typename V::memory_space, Kokkos::HostSpace> ||
        std::is_same_v<typename V::memory_space, Kokkos::AnonymousSpace>,
        "Can only convert a StridedMemRefType to a Kokkos::View in HostSpace.");
    if constexpr(std::is_same_v<Layout, Kokkos::LayoutStride>)
    {
      size_t extents[8] = {0};
      size_t strides[8] = {0};
      for(int i = 0; i < V::rank; i++) {
        extents[i] = smr.sizes[i];
        strides[i] = smr.strides[i];
      }
      Layout layout(
          extents[0], strides[0],
          extents[1], strides[1],
          extents[2], strides[2],
          extents[3], strides[3],
          extents[4], strides[4],
          extents[5], strides[5],
          extents[6], strides[6],
          extents[7], strides[7]);
      return V(&smr.data[smr.offset], layout);
    }
    size_t extents[8] = {
      KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
      KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX};
    for(int i = 0; i < V::rank; i++)
      extents[i] = smr.sizes[i];
    Layout layout(
        extents[0], extents[1], extents[2], extents[3],
        extents[4], extents[5], extents[6], extents[7]);
    if constexpr(std::is_same_v<Layout, Kokkos::LayoutLeft>)
    {
      int64_t expectedStride = 1;
      for(int i = 0; i < int(V::rank); i++)
      {
        if(expectedStride != smr.strides[i])
          Kokkos::abort("Cannot shallow-copy StridedMemRefType that is not contiguous and LayoutLeft to LayoutLeft Kokkos::View");
        expectedStride *= smr.sizes[i];
      }
    }
    else if constexpr(std::is_same_v<Layout, Kokkos::LayoutRight>)
    {
      int64_t expectedStride = 1;
      for(int i = int(V::rank) - 1; i >= 0; i--)
      {
        if(expectedStride != smr.strides[i])
          Kokkos::abort("Cannot shallow-copy StridedMemRefType that is not contiguous and LayoutRight to LayoutRight Kokkos::View");
        expectedStride *= smr.sizes[i];
      }
    }
    return V(&smr.data[smr.offset], layout);
  }

  // KeepAlive structure keeps a reference to Kokkos::Views which
  // are returned to Python. Since it's difficult to transfer ownership of a
  // Kokkos::View's memory to numpy, we just have the Kokkos::View maintain ownership
  // and return an unmanaged numpy array to Python.
  //
  // All these views will be deallocated during lapis_finalize to avoid leaking.
  // The downside is that if a function is called many times,
  // all its results are kept in memory at the same time.
  struct KeepAlive
  {
    virtual ~KeepAlive() {}
  };

  template<typename T>
  struct KeepAliveT : public KeepAlive
  {
    // Make a shallow-copy of val
    KeepAliveT(const T& val) : p(new T(val)) {}
    std::unique_ptr<T> p;
  };

  static std::vector<std::unique_ptr<KeepAlive>> alives;

  template<typename T>
  void keepAlive(const T& val)
  {
    alives.emplace_back(new KeepAliveT(val));
  }

  // DualView design
  // - DualView is a shallow object with a shared_ptr to a DualViewImpl.
  // - DualViewImpl has the actual host and device views as members
  //   - These may be managed or unmanaged
  // - DualViewImpl also has a shared_ptr reference to its "parent". This is another DualViewImpl (possibly of different type)
  //   that is considered the owner of the memory. The shared_ptr reference ensures
  //   - it stays alive as long as any child DualView is alive, even if the original parent declaration goes out of scope
  //   - it is deallocated as soon as the last child DualView goes out of scope
  // - Assume that any DualView's parent is contiguous, and can be deep-copied between h and d
  // - All DualViews with the same parent share the parent's modify flags
  //
  //  DualViewBase can also "keepAliveHost" to keep its host view alive until lapis_finalize is called.
  //  This is used to safely return host views to python for numpy arrays to alias.

  struct DualViewBase
  {
    virtual ~DualViewBase() {}
    virtual void syncHost() = 0;
    virtual void syncDevice() = 0;
    virtual void keepAliveHost() = 0;
    bool modified_host = false;
    bool modified_device = false;
    std::shared_ptr<DualViewBase> parent;

    void setParent(const std::shared_ptr<DualViewBase>& parent_)
    {
      this->parent = parent_;
    }
  };

  template<typename DataType, typename Layout>
    struct DualViewImpl : public DualViewBase
  {
    using HostView = Kokkos::View<DataType, Layout, Kokkos::DefaultHostExecutionSpace>;
    using DeviceView = Kokkos::View<DataType, Layout, Kokkos::DefaultExecutionSpace>;

    static constexpr bool deviceAccessesHost = Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::accessible;
    static constexpr bool hostAccessesDevice = Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, typename DeviceView::memory_space>::accessible;

    // Default constructor makes empty/non-allocated views
    DualViewImpl() : device_view(), host_view() {}

    // Constructor for allocating a new view.
    // Does not actually allocate anything yet; instead 
    DualViewImpl(
        const std::string& label,
        size_t ex0 = KOKKOS_INVALID_INDEX, size_t ex1 = KOKKOS_INVALID_INDEX, size_t ex2 = KOKKOS_INVALID_INDEX, size_t ex3 = KOKKOS_INVALID_INDEX,
        size_t ex4 = KOKKOS_INVALID_INDEX, size_t ex5 = KOKKOS_INVALID_INDEX, size_t ex6 = KOKKOS_INVALID_INDEX, size_t ex7 = KOKKOS_INVALID_INDEX)
    {
      if constexpr(hostAccessesDevice) {
        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_dev"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
        host_view = HostView(device_view.data(), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
      }
      else if constexpr(deviceAccessesHost) {
        // Otherwise, host_view must be a separate allocation.
        host_view = HostView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_host"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
        device_view = DeviceView(host_view.data(), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
      }
      else {
        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_dev"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
        host_view = HostView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_host"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
      }
    }

    // Constructor which is given explicit device and host views, and a parent.
    // This can be used for subviewing/casting operations.
    // Note: d,h should alias parent's memory, but they can
    // have a different data type and layout.
    DualViewImpl(DeviceView d, HostView h)
      : device_view(d), host_view(h) {}

    // Constructor taking a host or device view
    template<typename DT, typename... Args>
    DualViewImpl(Kokkos::View<DT, Args...> v)
    {
      using ViewType = decltype(v);
      using Space = typename ViewType::memory_space;
      if constexpr(std::is_same_v<typename DeviceView::memory_space, Space>) {
        // Treat v like a device view, even though it's possible that DeviceView and HostView have the same type.
        // In this case, the host view will alias it.
        modified_device = true;
        if constexpr(deviceAccessesHost) {
          host_view = HostView(v.data(), v.layout());
        }
        else {
          host_view = HostView(Kokkos::view_alloc(Kokkos::WithoutInitializing, v.label() + "_host"), v.layout());
        }
        device_view = v;
      }
      else {
        modified_host = true;
        if constexpr(deviceAccessesHost) {
          device_view = DeviceView(v.data(), v.layout());
        }
        else {
          device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, v.label() + "_dev"), v.layout());
        }
        host_view = v;
      }
    }

    void modifyHost()
    {
      parent->modified_host = true;
    }

    void modifyDevice()
    {
      parent->modified_device = true;
    }

    bool modifiedHost()
    {
      // note: parent may just point to this
      return parent->modified_host;
    }

    bool modifiedDevice()
    {
      // note: parent may just point to this
      return parent->modified_device;
    }

    void syncHost() override
    {
      if (device_view.data() == host_view.data()) {
        // Imitating Kokkos::DualView behavior: if device and host are the same space
        // then this sync (if required) is equivalent to a fence.
        if(parent->modified_device) {
          parent->modified_device = false;
          Kokkos::fence();
        }
      }
      else if (parent->modified_device) {
        if(parent.get() == this) {
          Kokkos::deep_copy(host_view, device_view);
          modified_device = false;
        }
        else {
          parent->syncHost();
        }
      }
    }

    void syncDevice() override
    {
      // If host and device views are the same, do not sync or fence
      // because all host execution spaces are synchronous.
      // Any changes on the host side are immediately visible on the device side.
      if (device_view.data() != host_view.data()) {
        if(parent.get() == this) {
          if(modified_host) {
            Kokkos::deep_copy(device_view, host_view);
            modified_host = false;
          }
        }
        else {
          parent->syncDevice();
        }
      }
    }

    void keepAliveHost() override
    {
      // keep the parent's host view alive.
      // It is assumed to be either managed,
      // or unmanaged but references memory (e.g. from numpy)
      // with a longer lifetime that any result from the current LAPIS function.
      keepAlive(host_view);
    }

    void deallocate() {
      device_view = DeviceView();
      host_view = HostView();
    }

    size_t extent(int dim) {
      return device_view.extent(dim);
    }

    size_t stride(int dim) {
      return device_view.stride(dim);
    }

    DeviceView device_view;
    HostView host_view;
  };

  template<typename DataType, typename Layout>
  struct DualView
  {
    using ImplType = DualViewImpl<DataType, Layout>;
    using DeviceView = typename ImplType::DeviceView;
    using HostView = typename ImplType::HostView;

    std::shared_ptr<ImplType> impl;
    bool syncHostWhenDestroyed = false;

    DualView() {
      impl = std::make_shared<ImplType>();
      // Even though no data is allocated, set impl's parent to itself
      // so that sync/modify calls are well defined
      impl->setParent(impl);
    }

    DualView(
        const std::string& label,
        size_t ex0 = KOKKOS_INVALID_INDEX, size_t ex1 = KOKKOS_INVALID_INDEX, size_t ex2 = KOKKOS_INVALID_INDEX, size_t ex3 = KOKKOS_INVALID_INDEX,
        size_t ex4 = KOKKOS_INVALID_INDEX, size_t ex5 = KOKKOS_INVALID_INDEX, size_t ex6 = KOKKOS_INVALID_INDEX, size_t ex7 = KOKKOS_INVALID_INDEX) {
      impl = std::make_shared<ImplType>(label, ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
      impl->setParent(impl);
    }

    template<typename V>
    DualView(const V& v) {
      static_assert(std::is_same_v<typename V::data_type, DataType>,
          "DualView constructor from view: data type must match exactly");
      impl = std::make_shared<ImplType>(v);
      impl->setParent(impl);
    }

    template<typename Parent>
    DualView(const DeviceView& d, const HostView& h, const Parent& parent) {
      impl = std::make_shared<ImplType>(d, h);
      // From the caller's point of view, parent is some DualView that is considered the parent of this.
      // But we need parent to point to the top-level parent, not just the immediate parent.
      // This way we don't have to pointer hop multiple times during sync/modify calls.
      impl->setParent(parent.impl->parent);
    }

    ~DualView() {
      if(!impl) return;
      if(syncHostWhenDestroyed) syncHost();
      DualViewBase* parent = impl->parent.get();
      impl.reset();
      // All DualViewBases keep a shared reference to themselves, so
      // parent always keeps a shared_ptr to itself. This would normally
      // prevent the parent destructor ever being called.
      //
      // So if parent now has a use count of 1, it is
      // only pointing to itself and no other DualView objects
      // are using it anymore.
      if(parent->parent.use_count() == 1) {
        // This will reset the last shared_ptr reference,
        // causing parent to be destroyed.
        parent->parent.reset();
      }
    }

    DeviceView device_view() const {
      return impl->device_view;
    }

    HostView host_view() const {
      return impl->host_view;
    }

    void modifyHost() {
      impl->parent->modified_host = true;
    }

    void modifyDevice() {
      impl->parent->modified_device = true;
    }

    [[nodiscard]] bool modifiedHost() const {
      // note: parent may just point to this
      return impl->parent->modified_host;
    }

    [[nodiscard]] bool modifiedDevice() const {
      // note: parent may just point to this
      return impl->parent->modified_device;
    }

    void syncHost() {
      impl->syncHost();
    }

    void syncDevice() {
      impl->syncDevice();
    }

    void deallocate() {
      // Default destructor of this will release reference to impl,
      // but this can also be used to explicitly release.
      impl.reset();
    }

    size_t extent(int dim) const {
      return impl->extent(dim);
    }

    size_t stride(int dim) const {
      return impl->stride(dim);
    }

    void keepAliveHost() const {
      impl->parent->keepAliveHost();
    }

    void syncHostOnDestroy() {
      syncHostWhenDestroyed = true;
    }
  };

  inline int threadParallelVectorLength(int par) {
    if (par < 1)
      return 1;
    int max_vector_length = TeamPolicy::vector_length_max();
    int vector_length = 1;
    while(vector_length < max_vector_length && vector_length * 6 < par) vector_length *= 2;
    return vector_length;
  }

#ifdef LAPIS_USE_KOKKOSKERNELS
  template<typename Shape, typename Rowptrs, typename Entries, typename AValues, typename XVector, typename YVector>
  void spmv(const Shape& shape, const Rowptrs& rowptrs, const Entries& entries, const AValues& avalues, const XVector& x, const YVector& y)
  {
    // Deduce types
    using Offset = typename Rowptrs::non_const_value_type;
    using Ordinal = typename Entries::non_const_value_type;
    using Scalar = typename AValues::non_const_value_type;
    using Device = Kokkos::DefaultExecutionSpace;
    using CrsMat = KokkosSparse::CrsMatrix<Scalar, Ordinal, Device, void, Offset>;
    Ordinal m = shape.m0[0];
    Ordinal n = shape.m0[1];
    Offset nnz = shape.m1[2];
    CrsMat A("A", m, n, nnz, avalues, rowptrs, entries);
    KokkosSparse::spmv("N", 1.0, A, x, 1.0, y);
  }

  template<typename AMatrix, typename BMatrix, typename CMatrix>
  void gemm(const AMatrix& a, const BMatrix& b, const CMatrix& c)
  {
    KokkosBlas::gemm("N", "N", 1.0, a, b, 1.0, c);
  }

  template<typename AMatrix, typename XVector, typename YVector>
  void gemv(const AMatrix& a, const XVector& x, const YVector& y)
  {
    KokkosBlas::gemv("N", 1.0, a, x, 1.0, y);
  }
#endif
} // namespace LAPIS

struct Struct0 {
  Struct0() = default;
  Struct0(const std::array<int64_t, 2>& m0_, const std::array<int64_t, 3>& m1_)
  : m0(m0_), m1(m1_) {}
  std::array<int64_t, 2> m0;
  std::array<int64_t, 3> m1;
};
LAPIS::DualView<double*, Kokkos::LayoutRight> spmv(LAPIS::DualView<int32_t*, Kokkos::LayoutRight> v1, LAPIS::DualView<int32_t*, Kokkos::LayoutRight> v2, LAPIS::DualView<double*, Kokkos::LayoutRight> v3, Struct0 v4, LAPIS::DualView<double*, Kokkos::LayoutRight> v5, LAPIS::DualView<double*, Kokkos::LayoutRight> v6);
extern "C" void lapis_initialize();
extern "C" void lapis_finalize();
LAPIS::DualView<double*, Kokkos::LayoutRight> spmv(LAPIS::DualView<int32_t*, Kokkos::LayoutRight> v1, LAPIS::DualView<int32_t*, Kokkos::LayoutRight> v2, LAPIS::DualView<double*, Kokkos::LayoutRight> v3, Struct0 v4, LAPIS::DualView<double*, Kokkos::LayoutRight> v5, LAPIS::DualView<double*, Kokkos::LayoutRight> v6) {
  auto v1_d = v1.device_view();
  auto v1_h = v1.host_view();
  auto v2_d = v2.device_view();
  auto v2_h = v2.host_view();
  auto v3_d = v3.device_view();
  auto v3_h = v3.host_view();
  auto v5_d = v5.device_view();
  auto v5_h = v5.host_view();
  auto v6_d = v6.device_view();
  auto v6_h = v6.host_view();
  int64_t v7 = v4.m1[2];
  size_t v8 = v7;
  ;
  Kokkos::View<double, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> v9_d_temp(v3.device_view().data());
  Kokkos::View<double, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace> v9_h_temp(v3.host_view().data());
  LAPIS::DualView<double, Kokkos::LayoutRight> v9(v9_d_temp, v9_h_temp, v3);
  auto v9_d = v9.device_view();
  auto v9_h = v9.host_view();
  Kokkos::LayoutStride v10_layout(v8, 1);
  Kokkos::View<double*, Kokkos::LayoutStride, Kokkos::DefaultHostExecutionSpace> v10_host(v9_h.data() + 0, v10_layout);
  Kokkos::View<double*, Kokkos::LayoutStride, Kokkos::DefaultExecutionSpace> v10_device(v9_d.data() + 0, v10_layout);
  LAPIS::DualView<double*, Kokkos::LayoutStride> v10(v10_device, v10_host, v9);
  ;
  auto v10_d = v10.device_view();
  auto v10_h = v10.host_view();
  int64_t v11 = v4.m0[0];
  size_t v12 = v11;
  ;
  int64_t v13 = v4.m1[0];
  size_t v14 = v13;
  ;
  Kokkos::View<int32_t, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> v15_d_temp(v1.device_view().data());
  Kokkos::View<int32_t, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace> v15_h_temp(v1.host_view().data());
  LAPIS::DualView<int32_t, Kokkos::LayoutRight> v15(v15_d_temp, v15_h_temp, v1);
  auto v15_d = v15.device_view();
  auto v15_h = v15.host_view();
  Kokkos::LayoutStride v16_layout(v14, 1);
  Kokkos::View<int32_t*, Kokkos::LayoutStride, Kokkos::DefaultHostExecutionSpace> v16_host(v15_h.data() + 0, v16_layout);
  Kokkos::View<int32_t*, Kokkos::LayoutStride, Kokkos::DefaultExecutionSpace> v16_device(v15_d.data() + 0, v16_layout);
  LAPIS::DualView<int32_t*, Kokkos::LayoutStride> v16(v16_device, v16_host, v15);
  ;
  auto v16_d = v16.device_view();
  auto v16_h = v16.host_view();
  int64_t v17 = v4.m1[1];
  size_t v18 = v17;
  ;
  Kokkos::View<int32_t, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> v19_d_temp(v2.device_view().data());
  Kokkos::View<int32_t, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace> v19_h_temp(v2.host_view().data());
  LAPIS::DualView<int32_t, Kokkos::LayoutRight> v19(v19_d_temp, v19_h_temp, v2);
  auto v19_d = v19.device_view();
  auto v19_h = v19.host_view();
  Kokkos::LayoutStride v20_layout(v18, 1);
  Kokkos::View<int32_t*, Kokkos::LayoutStride, Kokkos::DefaultHostExecutionSpace> v20_host(v19_h.data() + 0, v20_layout);
  Kokkos::View<int32_t*, Kokkos::LayoutStride, Kokkos::DefaultExecutionSpace> v20_device(v19_d.data() + 0, v20_layout);
  LAPIS::DualView<int32_t*, Kokkos::LayoutStride> v20(v20_device, v20_host, v19);
  ;
  auto v20_d = v20.device_view();
  auto v20_h = v20.host_view();
  v16.syncHost();
  int32_t v21 = v16_h(v12);
  size_t v22 = v21;
  ;
  v16.syncHost();
  int32_t v23 = v16_h(0);
  size_t v24 = v23;
  ;
  size_t v25 = v22 - v24;
  size_t v26 = v25 + v12;
  size_t v27 = v26 - 1;
  size_t v28 = v27 / v12;
  v20.syncDevice();
  v16.syncDevice();
  v10.syncDevice();
  v6.syncDevice();
  v6.modifyDevice();
  v5.syncDevice();
  auto lambda_v29 = 
  KOKKOS_LAMBDA(const LAPIS::TeamMember& team) {
    size_t induction = team.league_rank() * team.team_size() + team.team_rank();
    if(induction >= v12) return;
    double v30 = v6_d(induction);
    int32_t v31 = v16_d(induction);
    int64_t v32 = (int64_t) v31;
    size_t v33 = v32;
    ;
    size_t v34 = induction + 1;
    int32_t v35 = v16_d(v34);
    int64_t v36 = (int64_t) v35;
    size_t v37 = v36;
    ;
    size_t v38 = v37 - v33;
    double v39;
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, 0, v38),
    [=](size_t v40, double& lreduce1) {
      size_t v41 = v33 + v40;
      int32_t v42 = v20_d(v41);
      int64_t v43 = (int64_t) v42;
      size_t v44 = v43;
      ;
      double v45 = v10_d(v41);
      double v46 = v5_d(v44);
      double v47 = v45 * v46;
      double v48 = lreduce1 + v47;
      lreduce1 = v48;
      ;
    }, (v39));
    Kokkos::Sum<double> v39_joiner(v39);
    v39_joiner.join(v39, v30);
    ;
    Kokkos::single(Kokkos::PerThread(team), [&]() {
      v6_d(induction) = v39;
    });
  };
  int vectorLength_v49 = v28 ? LAPIS::threadParallelVectorLength(v28) : 8;
  size_t teamSize_v50 = 1;
  if constexpr(!std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>) {
  teamSize_v50 = LAPIS::TeamPolicy(1, 1, vectorLength_v49).team_size_recommended(lambda_v29, Kokkos::ParallelForTag{});
  }
  size_t leagueSize_v51 = (v12 + teamSize_v50 - 1) / teamSize_v50;
  Kokkos::parallel_for(LAPIS::TeamPolicy(leagueSize_v51, teamSize_v50, vectorLength_v49), lambda_v29);
  return v6;
}

extern "C" void py_spmv(StridedMemRefType<double, 1>** ret0, StridedMemRefType<int32_t, 1>* param0, StridedMemRefType<int32_t, 1>* param1, StridedMemRefType<double, 1>* param2, const Struct0& param3, StridedMemRefType<double, 1>* param4, StridedMemRefType<double, 1>* param5)
{
  auto param0_smr = LAPIS::stridedMemrefToView<Kokkos::View<int32_t*, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>>(*param0);
  auto param1_smr = LAPIS::stridedMemrefToView<Kokkos::View<int32_t*, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>>(*param1);
  auto param2_smr = LAPIS::stridedMemrefToView<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>>(*param2);
  auto param4_smr = LAPIS::stridedMemrefToView<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>>(*param4);
  auto param5_smr = LAPIS::stridedMemrefToView<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>>(*param5);
  auto results = spmv(param0_smr, param1_smr, param2_smr, param3, param4_smr, param5_smr);
  results.syncHost();
  **ret0 = LAPIS::viewToStridedMemref(results.host_view());
  LAPIS::keepAlive(results.host_view());
  if(results.host_view().data() == results.device_view().data()) 
    LAPIS::keepAlive(results.device_view());
  Kokkos::DefaultExecutionSpace().fence();
}


extern "C" void lapis_initialize() {
  if (!Kokkos::is_initialized()) Kokkos::initialize();
}

extern "C" void lapis_finalize()
{
  LAPIS::alives.clear();
  Kokkos::finalize();
}

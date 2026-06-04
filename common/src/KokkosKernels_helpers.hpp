// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOSKERNELS_HELPERS_HPP_
#define KOKKOSKERNELS_HELPERS_HPP_

#include "KokkosKernels_config.h"
#include "KokkosKernels_default_types.hpp"  // default_layout, default_memspace_t

#include <type_traits>

namespace KokkosKernels {
namespace Impl {

// Unify Layout of a View to PreferredLayoutType if possible
// (either matches already, or is rank-0/rank-1 and contiguous)
// Used to reduce number of code instantiations.
template <class ViewType, class PreferredLayoutType>
struct GetUnifiedLayoutPreferring {
  using array_layout =
      typename std::conditional<((ViewType::rank == 1) &&
                                 !std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutStride>) ||
                                    (ViewType::rank == 0),
                                PreferredLayoutType, typename ViewType::array_layout>::type;
};

template <class ViewType>
struct GetUnifiedLayout {
  using array_layout = typename GetUnifiedLayoutPreferring<ViewType, default_layout>::array_layout;
};

template <class T, class TX, bool do_const, bool isView = Kokkos::is_view<T>::value>
struct GetUnifiedScalarViewType {
  typedef typename TX::non_const_value_type type;
};

template <class T, class TX>
struct GetUnifiedScalarViewType<T, TX, false, true> {
  typedef Kokkos::View<
      typename T::non_const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayoutPreferring<T, typename TX::array_layout>::array_layout,
      typename T::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      type;
};

template <class T, class TX>
struct GetUnifiedScalarViewType<T, TX, true, true> {
  typedef Kokkos::View<
      typename T::const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayoutPreferring<T, typename TX::array_layout>::array_layout,
      typename T::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      type;
};

template <class execution_space, class original_device_type>
class GetUnifiedDeviceType {
  static_assert(Kokkos::is_execution_space_v<execution_space>,
      "GetUnifiedDeviceType requires its template argument to be a Kokkos execution space.");

  using preferred_memory_space = KokkosKernels::default_memspace_t<execution_space>;
  // This line provides extra caution if original_device_type uses a host-pinned memory space.
  // We avoid having a unified view pretend that its underlying host-pinned memory is
  // normal device memory. For example, Kokkos may use CUDA's _ldg intrinsic on CudaSpace and
  // CudaUVMSpace Views, but not CudaHostPinnedSpace.
  static constexpr bool compatible = std::is_same_v<execution_space, typename original_device_type::memory_space::execution_space>;
  using memory_space = std::conditional_t<compatible, preferred_memory_space, typename original_device_type::memory_space>;
public:
  using type = Kokkos::Device<execution_space, memory_space>;
};

// InternalView<...>::type gives the unmanaged View type to be used inside the unification layer.
// Its layout will be PreferredLayout if possible, and InputView::array_layout if not.
template <typename InputView, typename ExecSpace, typename PreferredLayout, bool constData, bool keepDevice = false>
class InternalView {
  using DataInternal =
      std::conditional_t<constData, typename InputView::const_data_type, typename InputView::non_const_data_type>;
  using LayoutInternal =
      typename GetUnifiedLayoutPreferring<InputView, PreferredLayout>::array_layout;
  using OriginalDevice = typename InputView::device_type;
  using DeviceInternal = std::conditional_t<keepDevice, OriginalDevice, typename GetUnifiedDeviceType<ExecSpace, OriginalDevice>::type>;
public:
  using type = Kokkos::View<DataInternal, LayoutInternal, DeviceInternal, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
};

template <typename InputView, typename ExecSpace, typename PreferredLayout, bool constData, bool keepDevice = false>
using InternalView_t = typename InternalView<InputView, ExecSpace, PreferredLayout, constData, keepDevice>::type;

// Get the internal version of a View for the unification layer.
// Internal can be determined using InternalView_t.
template <typename Internal, typename Input>
Internal unifyView(const Input& v) {
  // The unified device type may not always be directly 'assignable' from the input device type.
  // So first create a view with type Internal, but with the same device as Input.
  // Then we can finish constructing the unified view using a pointer and layout.
  using DataInternal   = typename Internal::data_type;
  using LayoutInternal = typename Internal::array_layout;
  using TempView =
      Kokkos::View<DataInternal, LayoutInternal, typename Input::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  TempView temp = v;
  return Internal(temp.data(), temp.layout());
}

template <class... Ts>
struct are_integral : std::bool_constant<((std::is_integral_v<Ts> || std::is_enum_v<Ts>)&&...)> {};

template <class... Ts>
inline constexpr bool are_integral_v = are_integral<Ts...>::value;

}  // namespace Impl
}  // namespace KokkosKernels
#endif

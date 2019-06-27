#include<gtest/gtest.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas1_iamax.hpp>
#include<KokkosKernels_TestUtils.hpp>

namespace Test {
  template<class ViewTypeA, class Device>
  void impl_test_iamax(int N) {

    typedef typename ViewTypeA::non_const_value_type ScalarA;
    typedef Kokkos::Details::ArithTraits<ScalarA> AT;
    typedef typename AT::mag_type mag_type;

    typedef Kokkos::View<ScalarA*[2],
       typename std::conditional<
                std::is_same<typename ViewTypeA::array_layout,Kokkos::LayoutStride>::value,
                Kokkos::LayoutRight, Kokkos::LayoutLeft>::type,Device> BaseTypeA;

    typedef typename BaseTypeA::size_type size_type;

    BaseTypeA b_a("A",N);

    ViewTypeA a = Kokkos::subview(b_a,Kokkos::ALL(),0);

    typename BaseTypeA::HostMirror h_b_a = Kokkos::create_mirror_view(b_a);

    typename ViewTypeA::HostMirror h_a = Kokkos::subview(h_b_a,Kokkos::ALL(),0);

    Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

    Kokkos::fill_random(b_a,rand_pool,ScalarA(10));

    Kokkos::fence();

    Kokkos::deep_copy(h_b_a,b_a);

    typename ViewTypeA::const_type c_a = a;

    mag_type expected_result = Kokkos::Details::ArithTraits<mag_type>::min();
    size_type expected_max_loc = 0;
    for(int i=0;i<N;i++) {      
      mag_type val = AT::abs(h_a(i));
      if(val > expected_result) { expected_result = val; expected_max_loc = i;}
    }
	
    if(N == 0) {expected_result = typename AT::mag_type(0); expected_max_loc = 0;}

    size_type nonconst_max_loc = KokkosBlas::iamax(a);
    ASSERT_EQ( nonconst_max_loc, expected_max_loc);

    size_type const_max_loc = KokkosBlas::iamax(c_a);
    ASSERT_EQ( const_max_loc, expected_max_loc);

  }

  template<class ViewTypeA, class Device>
  void impl_test_iamax_mv(int N, int K) {

    typedef typename ViewTypeA::non_const_value_type ScalarA;
    typedef Kokkos::Details::ArithTraits<ScalarA> AT;
    typedef typename AT::mag_type mag_type;
    typedef typename ViewTypeA::size_type size_type;

    typedef multivector_layout_adapter<ViewTypeA> vfA_type;

    typename vfA_type::BaseType b_a("A",N,K);

    ViewTypeA a = vfA_type::view(b_a);

    typedef multivector_layout_adapter<typename ViewTypeA::HostMirror> h_vfA_type;

    typename h_vfA_type::BaseType h_b_a = Kokkos::create_mirror_view(b_a);

    typename ViewTypeA::HostMirror h_a = h_vfA_type::view(h_b_a);

    Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);

    Kokkos::fill_random(b_a,rand_pool,ScalarA(10));

    Kokkos::fence();

    Kokkos::deep_copy(h_b_a,b_a);

    typename ViewTypeA::const_type c_a = a;

    mag_type* expected_result = new mag_type[K];
    size_type* expected_max_loc = new size_type[K];

    for(int j=0;j<K;j++) {
      expected_result[j] = Kokkos::Details::ArithTraits<mag_type>::min();
      for(int i=0;i<N;i++) {
        mag_type val = AT::abs(h_a(i,j));
        if(val > expected_result[j]) { expected_result[j] = val; expected_max_loc[j] = i;}
      }
      if(N == 0) {expected_result[j] = mag_type(0); expected_max_loc[j] = size_type(0);}
    }

    Kokkos::View<size_type*,Kokkos::HostSpace> r("Iamax::Result",K);

    KokkosBlas::iamax(r,a);
    for(int k=0;k<K;k++) {
      size_type nonconst_result = r(k);
      size_type exp_result = expected_max_loc[k];
      ASSERT_EQ( nonconst_result, exp_result);
    }

   /* KokkosBlas::iamax(r,c_a);
    for(int k=0;k<K;k++) {
      size_type const_result = r(k);
      size_type exp_result = expected_max_loc[k];
      ASSERT_EQ( const_result, exp_result);
    }
*/
    delete [] expected_result;
    delete [] expected_max_loc;
  }
}

template<class ScalarA, class Device>
int test_iamax() {

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutLeft, Device> view_type_a_ll;
  Test::impl_test_iamax<view_type_a_ll, Device>(0);
  Test::impl_test_iamax<view_type_a_ll, Device>(13);
  Test::impl_test_iamax<view_type_a_ll, Device>(1024);
  //Test::impl_test_iamax<view_type_a_ll, Device>(132231);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutRight, Device> view_type_a_lr;
  Test::impl_test_iamax<view_type_a_lr, Device>(0);
  Test::impl_test_iamax<view_type_a_lr, Device>(13);
  Test::impl_test_iamax<view_type_a_lr, Device>(1024);
  //Test::impl_test_iamax<view_type_a_lr, Device>(132231);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA*, Kokkos::LayoutStride, Device> view_type_a_ls;
  Test::impl_test_iamax<view_type_a_ls, Device>(0);
  Test::impl_test_iamax<view_type_a_ls, Device>(13);
  Test::impl_test_iamax<view_type_a_ls, Device>(1024);
  //Test::impl_test_iamax<view_type_a_ls, Device>(132231);
#endif

  return 1;
}

template<class ScalarA, class Device>
int test_iamax_mv() {

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  Test::impl_test_iamax_mv<view_type_a_ll, Device>(0,5);
  Test::impl_test_iamax_mv<view_type_a_ll, Device>(13,5);
  Test::impl_test_iamax_mv<view_type_a_ll, Device>(1024,5);
  //Test::impl_test_iamax_mv<view_type_a_ll, Device>(132231,5);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device> view_type_a_lr;
  Test::impl_test_iamax_mv<view_type_a_lr, Device>(0,5);
  Test::impl_test_iamax_mv<view_type_a_lr, Device>(13,5);
  Test::impl_test_iamax_mv<view_type_a_lr, Device>(1024,5);
  //Test::impl_test_iamax_mv<view_type_a_lr, Device>(132231,5);
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutStride, Device> view_type_a_ls;
  Test::impl_test_iamax_mv<view_type_a_ls, Device>(0,5);
  Test::impl_test_iamax_mv<view_type_a_ls, Device>(13,5);
  Test::impl_test_iamax_mv<view_type_a_ls, Device>(1024,5);
  //Test::impl_test_iamax_mv<view_type_a_ls, Device>(132231,5);
#endif

  return 1;}

#if defined(KOKKOSKERNELS_INST_FLOAT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, iamax_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::iamax_float");
    test_iamax<float,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
TEST_F( TestCategory, iamax_mv_float ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::iamax_mvfloat");
    test_iamax_mv<float,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, iamax_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::iamax_double");
    test_iamax<double,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
TEST_F( TestCategory, iamax_mv_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::iamax_mv_double");
    test_iamax_mv<double,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, iamax_complex_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::iamax_complex_double");
    test_iamax<Kokkos::complex<double>,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
TEST_F( TestCategory, iamax_mv_complex_double ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::iamax_mv_complex_double");
    test_iamax_mv<Kokkos::complex<double>,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_INT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, iamax_int ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::iamax_int");
    test_iamax<int,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
TEST_F( TestCategory, iamax_mv_int ) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::iamax_mv_int");
    test_iamax_mv<int,TestExecSpace> ();
  Kokkos::Profiling::popRegion();
}
#endif


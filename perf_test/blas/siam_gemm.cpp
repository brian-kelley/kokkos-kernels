#define LAPIS_USE_KOKKOSKERNELS
#include "gemm.hpp"
#include <Kokkos_Random.hpp>

int main()
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using DualV = LAPIS::DualView<float**, Kokkos::LayoutRight>;
  using DeviceV = typename DualV::DeviceView;
  using HostV = typename DualV::HostView;
  using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
  lapis_initialize();
  {
    ExecSpace().print_configuration(std::cout);
    RandPool pool(123);
    Kokkos::Timer t;
    int numTrials = 1000;
    //for(int n = 256; n <= 2048; n += 32)
    int n = 2048;
    {
      // Construct inputs
      DualV A("A", n, n);
      DualV B("B", n, n);
      DualV C("C", n, n);
      A.modifyDevice();
      Kokkos::fill_random(A.device_view(), pool, 0.0, 1.0);
      B.modifyDevice();
      Kokkos::fill_random(B.device_view(), pool, 0.0, 1.0);
      A.syncHost();
      B.syncHost();
      // Warmup
      forward(A, B, C);
      Kokkos::fence();
      t.reset();
      for(int i = 0; i < numTrials; i++) {
        forward(A, B, C);
        Kokkos::fence();
      }
      double elapsed = t.seconds();
      std::cout << "LAPIS: Square matrix with n = " << n << ": avg time = " << elapsed / numTrials << "\n";
    }
    {
      // Construct inputs
      Kokkos::View<float**> A("A", n, n);
      Kokkos::View<float**> B("B", n, n);
      Kokkos::View<float**> C("C", n, n);
      Kokkos::fill_random(A, pool, 0.0, 1.0);
      Kokkos::fill_random(B, pool, 0.0, 1.0);
      // Warmup
      KokkosBlas::gemm("N", "N", 1.0, A, B, 1.0, C);
      Kokkos::fence();
      t.reset();
      for(int i = 0; i < numTrials; i++) {
        KokkosBlas::gemm("N", "N", 1.0, A, B, 1.0, C);
        Kokkos::fence();
      }
      double elapsed = t.seconds();
      std::cout << "KK/Vendor: Square matrix with n = " << n << ": avg time = " << elapsed / numTrials << "\n";
    }
  }
  lapis_finalize();
  return 0;
}


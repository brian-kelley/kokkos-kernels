#include "Kokkos_Core.hpp"

#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_spgemm.hpp"
#include "KokkosSparse_spgemm_impl_new.hpp"

#include "KokkosKernels_Test_Structured_Matrix.hpp"

using Scalar  = default_scalar;
using Ordinal = default_lno_t;
using Offset  = default_size_type;
using Layout  = default_layout;

int main() {
  Kokkos::initialize();

  using device_type = typename Kokkos::Device<
      Kokkos::DefaultExecutionSpace,
      typename Kokkos::DefaultExecutionSpace::memory_space>;
  using execution_space = typename device_type::execution_space;
  using memory_space    = typename device_type::memory_space;
  using matrix_type =
      typename KokkosSparse::CrsMatrix<Scalar, Ordinal, device_type, void,
                                       Offset>;
  using rowmap_type = typename matrix_type::row_map_type::non_const_type;

  int return_value = 0;

  {
    // The mat_structure view is used to generate a matrix using
    // finite difference (FD) or finite element (FE) discretization
    // on a cartesian grid.
    // Each row corresponds to an axis (x, y and z)
    // In each row the first entry is the number of grid point in
    // that direction, the second and third entries are used to apply
    // BCs in that direction.
    Kokkos::View<Ordinal* [3], Kokkos::HostSpace> mat_structure(
        "Matrix Structure", 2);
    mat_structure(0, 0) = 10;  // Request 10 grid point in 'x' direction
    mat_structure(0, 1) = 1;   // Add BC to the left
    mat_structure(0, 2) = 1;   // Add BC to the right
    mat_structure(1, 0) = 10;  // Request 10 grid point in 'y' direction
    mat_structure(1, 1) = 1;   // Add BC to the bottom
    mat_structure(1, 2) = 1;   // Add BC to the top

    matrix_type A =
        Test::generate_structured_matrix2D<matrix_type>("FD", mat_structure);
    matrix_type B =
        Test::generate_structured_matrix2D<matrix_type>("FE", mat_structure);
    matrix_type C;

    // Create KokkosKernelHandle
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
        Offset, Ordinal, Scalar, execution_space, memory_space, memory_space>;
    KernelHandle kh;

    // Select an spgemm algorithm, limited by configuration at compile-time and
    // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
    // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
    std::string myalg("SPGEMM_KK_MEMORY");
    KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
        KokkosSparse::StringToSPGEMMAlgorithm(myalg);
    kh.create_spgemm_handle(spgemm_algorithm);
    rowmap_type correctRowmap(Kokkos::ViewAllocateWithoutInitializing("C rowmap (Correct)"), A.numRows() + 1);
    KokkosSparse::Experimental::spgemm_symbolic(&kh, A.numRows(), A.numCols(), B.numCols(), A.graph.row_map, A.graph.entries, false, B.graph.row_map, B.graph.entries, false, correctRowmap);
    //Compute my symbolic
    rowmap_type bmkRowmap(Kokkos::ViewAllocateWithoutInitializing("My rowmap"), A.numRows() + 1);
    KokkosSparse::Impl::bmk_SpGEMM_Symbolic(A.numRows(), A.numCols(), B.numCols(), &kh, A.graph.row_map, A.graph.entries, B.graph.row_map, B.graph.entries, bmkRowmap);
    std::cout << "My C rowmap:\n";
    KokkosKernels::Impl::print_1Dview(std::cout, bmkRowmap);
    std::cout << "Correct C rowmap:\n";
    KokkosKernels::Impl::print_1Dview(std::cout, correctRowmap);
  }

  Kokkos::finalize();

  return return_value;
}

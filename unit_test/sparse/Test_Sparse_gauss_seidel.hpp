/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include "KokkosKernels_Handle.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_spadd.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <vector>
#include "KokkosSparse_gauss_seidel.hpp"
#include "KokkosSparse_partitioning_impl.hpp"
#include "KokkosSparse_sor_sequential_impl.hpp"

namespace GSTest {

using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;
using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;

//Innermost function for testing:
//Run symbolic, numeric, and 2 apply sweeps.
//Then verify that res norm decreased.
//kh must already have a configured Gauss-Seidel handle.
//direction: 0 = symmetric, 1 = forward, 2 = backward
template <typename Handle, typename crsMat_t, typename vec_t>
void run_and_verify(
    Handle* kh,
    crsMat_t A,
    vec_t x,
    vec_t y,
    bool symmetric,
    int direction = 0)
{
  using lno_t = typename Handle::nnz_lno_t;
  using scalar_t = typename Handle::scalar_t;
  using KAT = Kokkos::ArithTraits<scalar_t>;
  using mag_t = typename KAT::mag_type;
  const scalar_t one = KAT::one();
  lno_t num_vecs = x.extent(1);
  EXPECT_EQ(num_vecs, y.extent(1));
  //Compute the norm of each y column
  //(initial norm, for zero starting solution)
  std::vector<mag_t> initial_norms;
  for(lno_t i = 0; i < num_vecs; i++)
  {
    //using abs to get a real number,
    //so it works if scalar_t is real or complex
    auto ycol = Kokkos::subview(y, Kokkos::ALL(), i);
    initial_norms.push_back(KAT::abs(KokkosBlas::dot(ycol, ycol)));
  }
  lno_t numRows = A.numRows();
  lno_t numCols = A.numCols();
  gauss_seidel_symbolic
    (kh, numRows, numCols, A.graph.row_map, A.graph.entries, symmetric);
  gauss_seidel_numeric
    (kh, numRows, numCols, A.graph.row_map, A.graph.entries, A.values, symmetric);
  const int iters = 2;
  switch (direction) {
  case 0:
    symmetric_gauss_seidel_apply
      (kh, numRows, numCols, A.graph.row_map, A.graph.entries, A.values, x, y, true, true, one, iters);
    break;
  case 1:
    forward_sweep_gauss_seidel_apply
      (kh, numRows, numCols, A.graph.row_map, A.graph.entries, A.values, x, y, true, true, one, iters);
    break;
  case 2:
    backward_sweep_gauss_seidel_apply
      (kh, numRows, numCols, A.graph.row_map, A.graph.entries, A.values, x, y, true, true, one, iters);
    break;
  default:
    throw std::logic_error("Logic error in test: direction should be between 0 and 2 inclusive");
  }
  //Now compute the new residuals using SPMV
  vec_t res(Kokkos::ViewAllocateWithoutInitializing("Residuals"), numRows, num_vecs);
  Kokkos::deep_copy(res, y);
  KokkosSparse::spmv("N", one, A, x, -one, res);
  for(lno_t i = 0; i < num_vecs; i++)
  {
    //using abs to get a real number,
    //so it works if scalar_t is real or complex
    auto resCol = Kokkos::subview(res, Kokkos::ALL(), i);
    mag_t resNorm = KAT::abs(KokkosBlas::dot(resCol, resCol));
    EXPECT_LT(resNorm, 0.5 * initial_norms[i]);
  }
}

template<typename vec_t>
vec_t create_x_vector(vec_t& kok_x, double max_value = 10.0) {
  typedef typename vec_t::value_type scalar_t;
  auto h_x = Kokkos::create_mirror_view (kok_x);
  for (size_t j = 0; j < h_x.extent(1); ++j){
    for (size_t i = 0; i < h_x.extent(0); ++i){
      scalar_t r =
          static_cast <scalar_t> (rand()) /
          static_cast <scalar_t> (RAND_MAX / max_value);
      h_x.access(i, j) = r;
    }
  }
  Kokkos::deep_copy (kok_x, h_x);
  return kok_x;
}

template <typename crsMat_t, typename vector_t>
vector_t create_y_vector(crsMat_t crsMat, vector_t x_vector){
  vector_t y_vector (Kokkos::ViewAllocateWithoutInitializing("Y VECTOR"),
      crsMat.numRows());
  KokkosSparse::spmv("N", 1, crsMat, x_vector, 0, y_vector);
  return y_vector;
}

//Create a strictly diag dominant linear system, with x as
//the correct solution. vec_t can be rank-1 or rank-2 view.
//A, x and y are all output arguments and don't need to
//be initialized or allocated.
template<typename crsMat_t, typename vec_t>
void create_problem(int numRows, int num_vecs, bool symmetric, crsMat_t& A, vec_t& x, vec_t& y)
{
  using size_type = typename crsMat_t::size_type;
  srand(234);
  //Add a few columns to represent ghosted entries:
  //important to test this for Ifpack2
  int numCols = numRows * 1.1;
  int nnzPerRow = 13;
  int nnzVariation = 4;
  size_type nnz = nnzPerRow * numRows;
  A = KokkosKernels::Impl::kk_generate_diagonally_dominant_sparse_matrix<crsMat_t>(numRows, numCols, nnz, nnzVariation, numRows / 10);
  if(symmetric)
  {
    //Symmetrize on host, rather than relying on the parallel versions (those can be tested for symmetric=false)
    crsMat_t A_trans = KokkosKernels::Impl::transpose_matrix(A);
    A = KokkosSparse::Experimental::spadd(A, A_trans);
  }
  //Create random LHS vector (x)
  x = vec_t(Kokkos::ViewAllocateWithoutInitializing("X"), A.numCols(), num_vecs);
  create_x_vector(x);
  //do a SPMV to find the RHS vector (y)
  y = create_y_vector(A, x);
}

template <typename scalar_t, typename lno_t, typename size_type, typename device, int rank>
void test_point(int numRows, bool symmetric)
{
  using mem_space = typename device::memory_space;
  using Handle = KokkosKernelsHandle<size_type, lno_t, scalar_t, typename device::execution_space, mem_space, mem_space>;
  using crsMat_t = KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type>;
  using vec_t = typename std::conditional<rank == 2,
        Kokkos::View<scalar_t**, mem_space>,
        Kokkos::View<scalar_t*, mem_space>>::type;
  int num_vecs = (vec_t::rank == 2) ? 3 : 1;
  crsMat_t A;
  vec_t x;
  vec_t y;
  create_problem(numRows, num_vecs, symmetric, A, x, y);
  //Just run for each apply direction
  //(there are no other options available for GS_POINT)
  for(int direction = 0; direction < 3; direction++)
  {
    Handle kh;
    kh.create_gs_handle(GS_POINT);
    run_and_verify<Handle, crsMat_t, vec_t>(&kh, A, x, y, symmetric, direction);
    kh.destroy_gs_handle();
  }
}

template <typename scalar_t, typename lno_t, typename size_type, typename device, int rank>
void test_cluster(int numRows, bool symmetric)
{
  using mem_space = typename device::memory_space;
  using Handle = KokkosKernelsHandle<size_type, lno_t, scalar_t, typename device::execution_space, mem_space, mem_space>;
  using crsMat_t = KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type>;
  using vec_t = typename std::conditional<rank == 2,
        Kokkos::View<scalar_t**, mem_space>,
        Kokkos::View<scalar_t*, mem_space>>::type;
  int num_vecs = (vec_t::rank == 2) ? 3 : 1;
  crsMat_t A;
  vec_t x;
  vec_t y;
  std::vector<int> clusterSizes = {2, 4, 19};
  create_problem(numRows, num_vecs, symmetric, A, x, y);
  for(int direction = 0; direction < 3; direction++)
  {
    for(int apply_algo = 0; apply_algo < (int) NUM_CGS_ALGORITHMS; apply_algo++)
    {
      for(int clusterSize : clusterSizes)
      {
        for(int mixed_prec = 0; mixed_prec < 2; mixed_prec++)
        {
          Handle kh;
          kh.create_gs_handle(apply_algo, CLUSTER_BALLOON, mixed_prec, clusterSize);
          run_and_verify<Handle, crsMat_t, vec_t>(&kh, A, x, y, symmetric, direction);
          kh.destroy_gs_handle();
        }
      }
    }
  }
}

//Test classic GS (sptrsv)
template <typename scalar_t, typename lno_t, typename size_type, typename device, int rank>
void test_classic(int numRows, bool symmetric)
{
  using mem_space = typename device::memory_space;
  using Handle = KokkosKernelsHandle<size_type, lno_t, scalar_t, typename device::execution_space, mem_space, mem_space>;
  using crsMat_t = KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type>;
  using vec_t = typename std::conditional<rank == 2,
        Kokkos::View<scalar_t**, mem_space>,
        Kokkos::View<scalar_t*, mem_space>>::type;
  int num_vecs = (vec_t::rank == 2) ? 3 : 1;
  crsMat_t A;
  vec_t x;
  vec_t y;
  std::vector<int> clusterSizes = {2, 4, 19};
  create_problem(numRows, num_vecs, symmetric, A, x, y);
  for(int direction = 0; direction < 3; direction++)
  {
    Handle kh;
    kh.create_gs_handle(GS_TWOSTAGE);
    kh.set_gs_twostage(false, numRows);
    run_and_verify<Handle, crsMat_t, vec_t>(&kh, A, x, y, symmetric, direction);
    kh.destroy_gs_handle();
  }
}

//Test two-stage GS+JR
template <typename scalar_t, typename lno_t, typename size_type, typename device, int rank>
void test_twostage(int numRows, bool symmetric)
{
  using mem_space = typename device::memory_space;
  using Handle = KokkosKernelsHandle<size_type, lno_t, scalar_t, typename device::execution_space, mem_space, mem_space>;
  using crsMat_t = KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type>;
  using vec_t = typename std::conditional<rank == 2,
        Kokkos::View<scalar_t**, mem_space>,
        Kokkos::View<scalar_t*, mem_space>>::type;
  int num_vecs = (vec_t::rank == 2) ? 3 : 1;
  crsMat_t A;
  vec_t x;
  vec_t y;
  std::vector<int> clusterSizes = {2, 4, 19};
  create_problem(numRows, num_vecs, symmetric, A, x, y);
  for(int direction = 0; direction < 3; direction++)
  {
    Handle kh;
    kh.create_gs_handle(GS_TWOSTAGE);
    kh.set_gs_twostage(true, numRows);
    run_and_verify<Handle, crsMat_t, vec_t>(&kh, A, x, y, symmetric, direction);
    kh.destroy_gs_handle();
  }
}

template <typename scalar_t, typename lno_t, typename size_type, typename device>
void test_sequential_sor(lno_t numRows, size_type nnz, lno_t bandwidth, lno_t row_size_variance) {
  const scalar_t zero = Kokkos::Details::ArithTraits<scalar_t>::zero();
  const scalar_t one = Kokkos::Details::ArithTraits<scalar_t>::one();
  srand(245);
  typedef typename device::execution_space exec_space;
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type> crsMat_t;
  lno_t numCols = numRows;
  crsMat_t input_mat = KokkosKernels::Impl::kk_generate_diagonally_dominant_sparse_matrix<crsMat_t>(numRows,numCols,nnz,row_size_variance, bandwidth);
  auto rowmap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), input_mat.graph.row_map);
  auto entries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), input_mat.graph.entries);
  auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), input_mat.values);
  //create raw x (unkown), y (rhs) vectors
  using vector_t = typename crsMat_t::values_type::non_const_type;
  //Create random x
  vector_t x("X", numRows);
  auto x_host = Kokkos::create_mirror_view(x);
  for(lno_t i = 0; i < numRows; i++)
  {
    x_host(i) = one * scalar_t(10.0 * rand() / RAND_MAX);
  }
  Kokkos::deep_copy(x, x_host);
  //record the correct solution, to compare against at the end
  vector_t xgold("X gold", numRows);
  Kokkos::deep_copy(xgold, x);
  vector_t y = create_y_vector(input_mat, x);
  exec_space().fence();
  auto y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  //initial solution is zero
  Kokkos::deep_copy(x_host, zero);
  //get the inverse diagonal (only needed on host)
  Kokkos::View<scalar_t*, Kokkos::HostSpace> invDiag("diag^-1", numRows);
  for(lno_t i = 0; i < numRows; i++)
  {
    for(size_type j = rowmap(i); j < rowmap(i + 1); j++)
    {
      if(entries(j) == i)
        invDiag(i) = one / values(j);
    }
  }
  for(int i = 0; i < 1; i++)
  {
    KokkosSparse::Impl::Sequential::gaussSeidel
      <lno_t, size_type, scalar_t, scalar_t, scalar_t>
      (numRows, 1, rowmap.data(), entries.data(), values.data(),
       y_host.data(), numRows,
       x_host.data(), numRows,
       invDiag.data(),
       one, //omega
       "F");
    KokkosSparse::Impl::Sequential::gaussSeidel
      <lno_t, size_type, scalar_t, scalar_t, scalar_t>
      (numRows, 1, rowmap.data(), entries.data(), values.data(),
       y_host.data(), numRows,
       x_host.data(), numRows,
       invDiag.data(),
       one, //omega
       "B");
  }
  //Copy solution back
  Kokkos::deep_copy(x, x_host);
  //Check against gold solution
  scalar_t xSq = KokkosBlas::dot(x, x);
  scalar_t solnDot = KokkosBlas::dot(x, xgold);
  double scaledSolutionDot = Kokkos::Details::ArithTraits<scalar_t>::abs(solnDot / xSq);
  EXPECT_TRUE(0.99 < scaledSolutionDot);
}

template <typename scalar_t, typename lno_t, typename size_type, typename device>
void test_balloon_clustering(lno_t numRows, size_type nnzPerRow, lno_t bandwidth)
{
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type> crsMat_t;
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type const_lno_row_view_t;
  typedef typename graph_t::entries_type const_lno_nnz_view_t;
  typedef typename graph_t::row_map_type::non_const_type lno_row_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef KokkosKernelsHandle
      <size_type, lno_t, scalar_t,
      typename device::execution_space, typename device::memory_space,typename device::memory_space> KernelHandle;
  srand(245);
  size_type nnzTotal = nnzPerRow * numRows;
  lno_t nnzVariance = nnzPerRow / 4;
  crsMat_t A = KokkosKernels::Impl::kk_generate_sparse_matrix<crsMat_t>(numRows, numRows, nnzTotal, nnzVariance, bandwidth);
  lno_row_view_t symRowmap;
  lno_nnz_view_t symEntries;
  KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap
    <const_lno_row_view_t, const_lno_nnz_view_t, lno_row_view_t, lno_nnz_view_t, typename device::execution_space>
    (numRows, A.graph.row_map, A.graph.entries, symRowmap, symEntries);
  KokkosSparse::Impl::BalloonClustering<KernelHandle, lno_row_view_t, lno_nnz_view_t> balloon(numRows, symRowmap, symEntries);
  for(int clusterSize = 1; clusterSize <= numRows / 16; clusterSize = std::ceil(clusterSize * 1.3))
  {
    auto vertClusters = balloon.run(clusterSize);
    //validate results: make sure cluster labels are in bounds, and that the number of clusters is correct
    auto vertClustersHost = Kokkos::create_mirror_view(vertClusters);
    Kokkos::deep_copy(vertClustersHost, vertClusters);
    lno_t numClusters = (numRows + clusterSize - 1) / clusterSize;
    //check the hard constraints of the clustering
    std::set<lno_t> uniqueClusterIDs;
    for(lno_t i = 0; i < numRows; i++)
    {
      EXPECT_TRUE(vertClustersHost(i) >= 0);
      EXPECT_TRUE(vertClustersHost(i) < numClusters);
      uniqueClusterIDs.insert(vertClustersHost(i));
    }
    EXPECT_TRUE(uniqueClusterIDs.size() == static_cast<size_t>(numClusters));
  }
}
}

#define EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE) \
TEST_F( TestCategory, sparse ## _ ## gauss_seidel_point ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  GSTest::test_point<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(2000, true); \
  GSTest::test_point<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(2000, false); \
  GSTest::test_point<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(0, false); \
  GSTest::test_point<SCALAR, ORDINAL, OFFSET, DEVICE, 2>(1000, false); \
} \
TEST_F( TestCategory, sparse ## _ ## gauss_seidel_cluster ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  GSTest::test_cluster<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(2000, true); \
  GSTest::test_cluster<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(2000, false); \
  GSTest::test_cluster<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(0, false); \
  GSTest::test_cluster<SCALAR, ORDINAL, OFFSET, DEVICE, 2>(1000, false); \
} \
TEST_F( TestCategory, sparse ## _ ## gauss_seidel_twostage ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  GSTest::test_twostage<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(2000, true); \
  GSTest::test_twostage<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(2000, false); \
  GSTest::test_twostage<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(0, false); \
  GSTest::test_twostage<SCALAR, ORDINAL, OFFSET, DEVICE, 2>(1000, false); \
} \
TEST_F( TestCategory, sparse ## _ ## gauss_seidel_classic ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  GSTest::test_classic<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(2000, true); \
  GSTest::test_classic<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(2000, false); \
  GSTest::test_classic<SCALAR, ORDINAL, OFFSET, DEVICE, 1>(0, false); \
  GSTest::test_classic<SCALAR, ORDINAL, OFFSET, DEVICE, 2>(1000, false); \
} \
TEST_F( TestCategory, sparse ## _ ## balloon_clustering ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  GSTest::test_balloon_clustering<SCALAR,ORDINAL,OFFSET,DEVICE>(5000, 100, 2000); \
} \
TEST_F( TestCategory, sparse ## _ ## sequential_sor ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  GSTest::test_sequential_sor<SCALAR,ORDINAL,OFFSET,DEVICE>(1000, 1000 * 15, 50, 10); \
}

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int64_t, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int64_t, size_t, TestExecSpace)
#endif


#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(Kokkos::complex<double>, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(Kokkos::complex<double>, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(Kokkos::complex<double>, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(Kokkos::complex<double>, int64_t, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(Kokkos::complex<float>, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(Kokkos::complex<float>, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(Kokkos::complex<float>, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(Kokkos::complex<float>, int64_t, size_t, TestExecSpace)
#endif


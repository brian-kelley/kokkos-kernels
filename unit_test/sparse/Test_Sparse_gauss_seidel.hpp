/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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
//#include <Kokkos_Sparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <cstdlib>
#include <iostream>
#include <complex>
#include "KokkosSparse_gauss_seidel.hpp"
#include "KokkosSparse_rcm_impl.hpp"

#ifndef kokkos_complex_double
#define kokkos_complex_double Kokkos::complex<double>
#define kokkos_complex_float Kokkos::complex<float>
#endif

using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;
using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
namespace Test {

template <typename crsMat_t, typename device>
int run_gauss_seidel_1(
    crsMat_t input_mat,
    KokkosSparse::GSAlgorithm gs_algorithm,
    typename crsMat_t::values_type::non_const_type x_vector,
    typename crsMat_t::values_type::const_type y_vector,
    bool is_symmetric_graph,
    int apply_type = 0, // 0 for symmetric, 1 for forward, 2 for backward.
    bool skip_symbolic = false,
    bool skip_numeric = false,
    typename crsMat_t::value_type omega = Kokkos::Details::ArithTraits<typename crsMat_t::value_type>::one()
    ){
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type lno_view_t;
  typedef typename graph_t::entries_type   lno_nnz_view_t;
  typedef typename crsMat_t::values_type::non_const_type scalar_view_t;



  typedef typename lno_view_t::value_type size_type;
  typedef typename lno_nnz_view_t::value_type lno_t;
  typedef typename scalar_view_t::value_type scalar_t;

  typedef KokkosKernelsHandle
      <size_type,lno_t, scalar_t,
      typename device::execution_space, typename device::memory_space,typename device::memory_space > KernelHandle;



  KernelHandle kh;
  kh.set_team_work_size(16);
  kh.set_dynamic_scheduling(true);
  kh.create_gs_handle(gs_algorithm);


  const size_t num_rows_1 = input_mat.numRows();
  const size_t num_cols_1 = input_mat.numCols();
  const int apply_count = 100;

  if (!skip_symbolic){
    gauss_seidel_symbolic
      (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, is_symmetric_graph);
  }

  if (!skip_numeric){
    gauss_seidel_numeric
    (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, input_mat.values, is_symmetric_graph);
  }

  switch (apply_type){
  case 0:
    symmetric_gauss_seidel_apply
      (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, input_mat.values, x_vector, y_vector,false, true, omega, apply_count);
    break;
  case 1:
    forward_sweep_gauss_seidel_apply
    (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, input_mat.values, x_vector, y_vector,false, true, omega, apply_count);
    break;
  case 2:
    backward_sweep_gauss_seidel_apply
    (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, input_mat.values, x_vector, y_vector,false, true, omega, apply_count);
    break;
  default:
    symmetric_gauss_seidel_apply
    (&kh, num_rows_1, num_cols_1, input_mat.graph.row_map, input_mat.graph.entries, input_mat.values, x_vector, y_vector,false, true, omega, apply_count);
    break;
  }


  kh.destroy_gs_handle();
  return 0;
}

template<typename scalar_view_t>
scalar_view_t create_x_vector(size_t nv, double max_value = 10.0){
  scalar_view_t kok_x ("X", nv);


  typename scalar_view_t::HostMirror h_x =  Kokkos::create_mirror_view (kok_x);


  for (size_t i = 0; i < nv; ++i){
    typename scalar_view_t::value_type r =
        static_cast <typename scalar_view_t::value_type> (rand()) /
        static_cast <typename scalar_view_t::value_type> (RAND_MAX / max_value);
    h_x(i) = r;
    //h_x(i) = 1;
  }
  Kokkos::deep_copy (kok_x, h_x);


  return kok_x;
}
template <typename crsMat_t, typename vector_t>
vector_t create_y_vector(crsMat_t crsMat, vector_t x_vector){
  vector_t y_vector ("Y VECTOR", crsMat.numRows());
  KokkosSparse::spmv("N", 1, crsMat, x_vector, 1, y_vector);
  return y_vector;
}

}

template <typename scalar_t, typename lno_t, typename size_type, typename device>
void test_gauss_seidel(lno_t numRows, size_type nnz, lno_t bandwidth, lno_t row_size_variance) {

  using namespace Test;
  srand(245);
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type> crsMat_t;
  //typedef typename crsMat_t::StaticCrsGraphType graph_t;
  //typedef typename graph_t::row_map_type lno_view_t;
  //typedef typename graph_t::entries_type lno_nnz_view_t;
  //typedef typename graph_t::entries_type::non_const_type   color_view_t;
  typedef typename crsMat_t::values_type::non_const_type scalar_view_t;

  lno_t numCols = numRows;
  crsMat_t input_mat = KokkosKernels::Impl::kk_generate_diagonally_dominant_sparse_matrix<crsMat_t>(numRows,numCols,nnz,row_size_variance, bandwidth);

  lno_t nv = input_mat.numRows();

  //KokkosKernels::Impl::print_1Dview(input_mat.graph.row_map);
  //KokkosKernels::Impl::print_1Dview(input_mat.graph.entries);
  //KokkosKernels::Impl::print_1Dview(input_mat.values);

  //scalar_view_t solution_x ("sol", nv);
  //Kokkos::Random_XorShift64_Pool<ExecutionSpace> g(1931);
  //Kokkos::fill_random(solution_x,g,Kokkos::Random_XorShift64_Pool<ExecutionSpace>::generator_type::MAX_URAND);

  const scalar_view_t solution_x = create_x_vector<scalar_view_t>(nv);
  scalar_view_t y_vector = create_y_vector(input_mat, solution_x);
#ifdef gauss_seidel_testmore
  GSAlgorithm gs_algorithms[] ={GS_DEFAULT, GS_TEAM, GS_PERMUTED};
  int apply_count = 3;
  for (int ii = 0; ii < 3; ++ii){
#else
  int apply_count = 1;
  GSAlgorithm gs_algorithms[] ={GS_DEFAULT};
  for (int ii = 0; ii < 1; ++ii){
#endif
    GSAlgorithm gs_algorithm = gs_algorithms[ii];
    scalar_view_t x_vector ("x vector", nv);
    const scalar_t alpha = 1.0;
    KokkosBlas::axpby(alpha, solution_x, -alpha, x_vector);
    scalar_t dot_product = KokkosBlas::dot( x_vector , x_vector );
    typedef typename Kokkos::Details::ArithTraits<scalar_t>::mag_type mag_t;
    mag_t initial_norm_res = Kokkos::Details::ArithTraits<scalar_t>::abs (dot_product);
    initial_norm_res  = Kokkos::Details::ArithTraits<mag_t>::sqrt( initial_norm_res );
    Kokkos::deep_copy (x_vector , 0);

    //bool is_symmetric_graph = false;
    //int apply_type = 0;
    //bool skip_symbolic = false;
    //bool skip_numeric = false;
    scalar_t omega = 0.9;

    for (int is_symmetric_graph = 0; is_symmetric_graph < 2; ++is_symmetric_graph){

      for (int apply_type = 0; apply_type < apply_count; ++apply_type){
        for (int skip_symbolic = 0; skip_symbolic < 2; ++skip_symbolic){
          for (int skip_numeric = 0; skip_numeric < 2; ++skip_numeric){

            Kokkos::Impl::Timer timer1;
            //int res =
            run_gauss_seidel_1<crsMat_t, device>(input_mat, gs_algorithm, x_vector, y_vector, is_symmetric_graph, apply_type, skip_symbolic, skip_numeric, omega);
            //double gs = timer1.seconds();

            //KokkosKernels::Impl::print_1Dview(x_vector);
            KokkosBlas::axpby(alpha, solution_x, -alpha, x_vector);
            //KokkosKernels::Impl::print_1Dview(x_vector);
            scalar_t result_dot_product = KokkosBlas::dot( x_vector , x_vector );
            mag_t result_norm_res  = Kokkos::Details::ArithTraits<scalar_t>::abs( result_dot_product );
            result_norm_res = Kokkos::Details::ArithTraits<mag_t>::sqrt(result_norm_res);
            //std::cout << "result_norm_res:" << result_norm_res << " initial_norm_res:" << initial_norm_res << std::endl;
            EXPECT_TRUE( (result_norm_res < initial_norm_res));
          }
        }
      }
    }
  }
  //device::execution_space::finalize();
}

//Generate a symmetric, diagonally dominant matrix for testing RCM
template<typename crsMat_t, typename scalar_t, typename lno_t, typename device, typename size_type>
crsMat_t genSymmetricMatrix(lno_t numRows, lno_t randNNZ, lno_t bandwidth, bool isGraphConnected)
{
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type rowmap_view;
  typedef typename graph_t::entries_type::non_const_type colinds_view;
  typedef typename crsMat_t::values_type::non_const_type scalar_view;
  std::vector<bool> dense(numRows * numRows, false);
  auto addBand = [&](int n)
  {
    for(int i = 0; i < numRows; i++)
    {
      if(i + n < numRows)
      {
        dense[i + n + i * numRows] = true;
        dense[i + (i + n) * numRows] = true;
      }
      if(i - n >= 0)
      {
        dense[i - n + i * numRows] = true;
        dense[i + (i - n) * numRows] = true;
      }
    }
  };
  //add diagonal and another band
  addBand(0);
  //add random edges
  for(lno_t i = 0; i < randNNZ; i++)
  {
    int row = rand() % numRows;
    int col = rand() % numRows;
    dense[row + col * numRows] = true;
    dense[col + row * numRows] = true;
  }
  if(isGraphConnected)
  {
    //Add a minimum set of edges to make graph connected
    std::set<lno_t> connected;
    for(lno_t i = 0; i < numRows; i++)
    {
      if(dense[i])
        connected.insert(i);
    }
    while((lno_t) connected.size() < numRows)
    {
      lno_t toConnect = 0;
      for(; toConnect < numRows; toConnect++)
      {
        if(connected.find(toConnect) == connected.end())
        {
          break;
        }
      }
      size_t index = connected.size() - 1 - rand() % connected.size();
      lno_t inGraph;
      for(auto c : connected)
      {
        if(index == 0)
        {
          inGraph = c;
          break;
        }
        index--;
      }
      dense[toConnect + inGraph * numRows] = true;
      dense[inGraph + toConnect * numRows] = true;
      for(lno_t i = 0; i < numRows; i++)
      {
        if(dense[toConnect + i * numRows])
          connected.insert(i);
      }
    }
  }
  /*
  std::cout << "Graph for testing RCM:\n";
  for(int i = 0; i < numRows; i++)
  {
    for(int j = 0; j < numRows; j++)
    {
      if(dense[i * numRows + j])
        std::cout << '*';
      else
        std::cout << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
  */
  size_t nnz = std::count_if(dense.begin(), dense.end(), [](bool v) {return v;});
  rowmap_view Arowmap("asdf", numRows + 1);
  colinds_view Acolinds("asdf", nnz);
  scalar_view Avalues("asdf", nnz);
  size_t total = 0;
  for(lno_t i = 0; i < numRows; i++)
  {
    Arowmap(i) = total;
    for(lno_t j = 0; j < numRows; j++)
    {
      if(dense[i * numRows + j])
      {
        Acolinds(total) = j;
        if(i == j)
          Avalues(total) = 1000;
        else
          Avalues(total) = 1;
        total++;
      }
    }
  }
  Arowmap(numRows) = total;
  //std::cout << "Actual NNZ in matrix: " << total << '\n';
  //std::cout << "But, allocated space for : " << Avalues.dimension_0() << '\n';
  graph_t Agraph(Acolinds, Arowmap);
  return crsMat_t("RCM test matrix", numRows, Avalues, Agraph);
}

template<typename Vector>
double norm2(Vector& v)
{
  double sqSum = 0;
  for(size_t i = 0; i < v.dimension_0(); i++)
  {
    sqSum += v(i) * v(i);
  }
  return sqrt(sqSum);
}

template <typename scalar_t, typename lno_t, typename offset_t, typename device>
void test_cluster_convergence(std::string matrixPath)
{
  using namespace Test;
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, offset_t> crsMat_t;
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename crsMat_t::values_type::non_const_type scalar_view_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename lno_view_t::size_type size_type;
  typedef KokkosKernelsHandle
      <offset_t, lno_t, scalar_t,
      typename device::execution_space, typename device::memory_space,typename device::memory_space> KernelHandle;
  srand(245);
  lno_view_t Arowmap("A rowmap", 0);
  lno_nnz_view_t Aentries("A entries", 0);
  scalar_view_t Avalues("A values", 0);
  lno_t numRows = 0;
  offset_t nnz = 0;
  std::ofstream output("cluster_convergence.txt");
  output << "Loading matrix \"" << matrixPath << "...";
  {
    offset_t* rowmap = nullptr;
    lno_t* entries = nullptr;
    scalar_t* values = nullptr;
    KokkosKernels::Impl::read_mtx<lno_t, offset_t, scalar_t>(matrixPath.c_str(), &numRows, &nnz, &rowmap, &entries, &values, false, false, false);
    output << "done.\n";
    output << "Loaded matrix with " << numRows << " rows and " << nnz << " entries.\n";
    //now that values have been read in, copy them to the actual views
    Kokkos::resize(Arowmap, numRows + 1);
    Kokkos::resize(Aentries, nnz);
    Kokkos::resize(Avalues, nnz);
    auto Arowmap_host = Kokkos::create_mirror_view(Arowmap);
    auto Aentries_host = Kokkos::create_mirror_view(Aentries);
    auto Avalues_host = Kokkos::create_mirror_view(Avalues);
    memcpy(Arowmap_host.data(), rowmap, (numRows + 1) * sizeof(offset_t));
    memcpy(Aentries_host.data(), entries, nnz * sizeof(lno_t));
    memcpy(Avalues_host.data(), values, nnz * sizeof(scalar_t));
    Kokkos::deep_copy(Arowmap, Arowmap_host);
    Kokkos::deep_copy(Aentries, Aentries_host);
    Kokkos::deep_copy(Avalues, Avalues_host);
    free(rowmap);
    free(entries);
    free(values);
  }
  crsMat_t A("A", numRows, numRows, nnz, Avalues, Arowmap, Aentries);
  //create a randomized RHS vector (b)
  scalar_view_t b("b", numRows);
  for(lno_t i = 0; i < numRows; i++)
  {
    b(i) = (double) rand() / RAND_MAX;
  }
  //create solution vector (x), zero initial guess
  //try a bunch of powers of 2 for cluster sizes
  std::vector<lno_t> clusterSizes;
  for(size_type i = 1; i <= 8192; i <<= 1)
    clusterSizes.push_back(i);
  const int niters = 10;
  double bnorm = norm2(b);
  std::cout << "Norm of b: " << bnorm << '\n';
  for(size_t test = 0; test < clusterSizes.size(); test++)
  {
    auto clusterSize = clusterSizes[test];
    output << "Testing cluster size = " << clusterSize << '\n';
    //starting solution is zero vector
    scalar_view_t x("x", numRows);
    KernelHandle kh;
    kh.create_gs_handle(clusterSize);
    //only need to do G-S setup (symbolic/numeric) once
    Kokkos::Impl::Timer timer;
    KokkosSparse::Experimental::gauss_seidel_symbolic<KernelHandle, lno_view_t, lno_nnz_view_t>
      (&kh, numRows, numRows, Arowmap, Aentries, true);
    KokkosSparse::Experimental::gauss_seidel_numeric<KernelHandle, lno_view_t, lno_nnz_view_t, scalar_view_t>
      (&kh, numRows, numRows, Arowmap, Aentries, Avalues, true);
    output << "Cluster size " << clusterSize << " setup time: " << timer.seconds() << " s\n";
    timer.reset();
    KokkosSparse::Experimental::symmetric_gauss_seidel_apply
      <KernelHandle, lno_view_t, lno_nnz_view_t, scalar_view_t, scalar_view_t, scalar_view_t>
      (&kh, numRows, numRows, Arowmap, Aentries, Avalues, x, b, false, true, niters);
    output << "Cluster size " << clusterSize << " apply time: " << timer.seconds() << " s\n";
    scalar_view_t res("Ax-b", numRows);
    Kokkos::deep_copy(res, b);
    scalar_t alpha = 1;
    scalar_t beta = -1;
    KokkosSparse::spmv<scalar_t, crsMat_t, scalar_view_t, scalar_t, scalar_view_t>
      ("N", alpha, A, x, beta, res, KokkosSparse::RANK_ONE());
    scalar_t norm = norm2(res);
    output << "Cluster size " << clusterSize << " norm after " << niters << " sweeps: " << norm << '\n';
    output << "Cluster size " << clusterSize << " proportion of residual eliminated: " << 1.0 - (norm / bnorm) << std::endl;
  }
  output.close();
}

template <typename scalar_t, typename lno_t, typename offset_t, typename device>
void test_rcm_perf(std::string matrixPath)
{
  using namespace Test;
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, offset_t> crsMat_t;
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename crsMat_t::values_type::non_const_type scalar_view_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename lno_view_t::size_type size_type;
  typedef KokkosKernelsHandle
      <offset_t, lno_t, scalar_t,
      typename device::execution_space, typename device::memory_space,typename device::memory_space> KernelHandle;
  KernelHandle kh;
  srand(245);
  lno_view_t Arowmap("A rowmap", 0);
  lno_nnz_view_t Aentries("A entries", 0);
  scalar_view_t Avalues("A values", 0);
  lno_t numRows = 0;
  offset_t nnz = 0;
  {
    offset_t* rowmap = nullptr;
    lno_t* entries = nullptr;
    scalar_t* values = nullptr;
    KokkosKernels::Impl::read_mtx<lno_t, offset_t, scalar_t>(matrixPath.c_str(), &numRows, &nnz, &rowmap, &entries, &values, false, false, false);
    //now that values have been read in, copy them to the actual views
    Kokkos::resize(Arowmap, numRows + 1);
    Kokkos::resize(Aentries, nnz);
    Kokkos::resize(Avalues, nnz);
    auto Arowmap_host = Kokkos::create_mirror_view(Arowmap);
    auto Aentries_host = Kokkos::create_mirror_view(Aentries);
    auto Avalues_host = Kokkos::create_mirror_view(Avalues);
    memcpy(Arowmap_host.data(), rowmap, (numRows + 1) * sizeof(offset_t));
    memcpy(Aentries_host.data(), entries, nnz * sizeof(lno_t));
    memcpy(Avalues_host.data(), values, nnz * sizeof(scalar_t));
    Kokkos::deep_copy(Arowmap, Arowmap_host);
    Kokkos::deep_copy(Aentries, Aentries_host);
    Kokkos::deep_copy(Avalues, Avalues_host);
    free(rowmap);
    free(entries);
    free(values);
  }
  Kokkos::Impl::Timer timer;
  int trials = 20;
  for(int i = 0; i < trials; i++)
  {
    KokkosSparse::Impl::RCM<KernelHandle, lno_view_t, lno_nnz_view_t> rcm(&kh, numRows, Arowmap, Aentries);
    rcm.rcm();
  }
  std::cout << "RCM (" << trials << " trials, " << numRows << " rows) took " << timer.seconds() / trials << " s (avg)\n";
}

template <typename scalar_t, typename lno_t, typename offset_t, typename device>
void test_gc_crash(std::string graphPath)
{
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, offset_t> crsMat_t;
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename crsMat_t::values_type::non_const_type scalar_view_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename lno_view_t::size_type size_type;
  lno_t nrows;
  offset_t numEntries; 
  offset_t* xadjRaw;
  lno_t* adjRaw;
  scalar_t* scalarRaw;
  KokkosKernels::Impl::read_graph_crs<lno_t, offset_t, scalar_t>
    (&nrows, &numEntries, &xadjRaw, &adjRaw, &scalarRaw, graphPath.c_str());
  lno_view_t xadj("xadj", nrows + 1);
  for(size_type i = 0; i < nrows + 1; i++)
    xadj(i) = xadjRaw[i];
  lno_view_t adj("adj", numEntries);
  for(size_type i = 0; i < numEntries; i++)
    adj(i) = adjRaw[i];
  std::cout << "Read in CRS graph with " << nrows << " rows and " << numEntries << " entries.\n";
  typedef KokkosKernelsHandle
      <offset_t, lno_t, scalar_t,
      typename device::execution_space, typename device::memory_space,typename device::memory_space> KernelHandle;
  KernelHandle kh;
  kh.create_graph_coloring_handle();
  std::cout << "Calling graph color kernel on cluster test graph..." << std::endl;
  KokkosGraph::Experimental::graph_color_symbolic(&kh, nrows, nrows, xadj, adj, false); 
  auto coloringHandle = kh.get_graph_coloring_handle();
  auto clusterColors = coloringHandle->get_vertex_colors();
  auto numClusterColors = coloringHandle->get_num_colors();
  std::cout << "Success, used " << numClusterColors << " colors." << std::endl;
}


template <typename scalar_t, typename lno_t, typename offset_t, typename device>
void test_rcm(lno_t numRows, offset_t nnz, offset_t bandwidth)
{
  using namespace Test;
  srand(245);
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, offset_t> crsMat_t;
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename lno_view_t::size_type size_type;
  //typedef typename graph_t::entries_type::non_const_type   color_view_t;
  typedef KokkosKernelsHandle
      <offset_t, lno_t, scalar_t,
      typename device::execution_space, typename device::memory_space,typename device::memory_space> KernelHandle;

  //std::cout << "building RCM test matrix with " << numRows << " rows.\n";
  crsMat_t A = genSymmetricMatrix<crsMat_t, scalar_t, lno_t, device, offset_t>(numRows, nnz, bandwidth, true);

  lno_view_t Arowmap("asdf", numRows + 1);
  nnz = A.graph.row_map(numRows);
  lno_nnz_view_t Aentries("asdf", nnz);
  lno_nnz_view_t Avalues("asdf", nnz);
  Kokkos::deep_copy(Arowmap, A.graph.row_map);
  //std::cout << "Aentries dim: " << Aentries.dimension_0() << '\n';
  //std::cout << "A.graph.entries dim: " << A.graph.entries.dimension_0() << '\n';
  for(offset_t i = 0; i < nnz; i++)
  {
    Aentries(i) = A.graph.entries(i);
    Avalues(i) = A.values(i);
  }
  Kokkos::deep_copy(Aentries, A.graph.entries);
  Kokkos::deep_copy(Avalues, A.values);

  KernelHandle kh;
  kh.create_gs_handle(GS_DEFAULT);

  typedef KokkosSparse::Impl::RCM<KernelHandle, decltype(Arowmap), decltype(Aentries)> rcm_t;
  rcm_t rcm(&kh, numRows, Arowmap, Aentries);
  /*&
  std::cout << "Matrix for RCM testing (raw CRS)\n";
  for(lno_t i = 0; i < numRows; i++)
  {
    std::cout << "Row " << i << ": ";
    for(offset_t j = Arowmap(i); j < Arowmap(i + 1); j++)
    {
      std::cout << Aentries(j) << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
  */
  //rcmOrder(i) = the timestamp of node i
  auto rcmOrder = rcm.rcm();
  //perm(i) = the node with timestamp i
  /*
  std::cout << "RCM row list that was returned:\n";
  for(size_type i = 0; i < rcmOrder.dimension_0(); i++)
  {
    std::cout << rcmOrder(i) << ' ';
  }
  std::cout << '\n';
  */
  /*
  std::cout << "Permutation array:\n";
  for(size_type i = 0; i < perm.dimension_0(); i++)
  {
    std::cout << perm(i) << ' ';
  }
  std::cout << '\n';
  */
  //make sure that perm is in fact a permuation matrix (contains each row exactly once)
  std::set<lno_t> rowSet;
  for(lno_t i = 0; i < numRows; i++)
    rowSet.insert(rcmOrder(i));
  if((lno_t) rowSet.size() != numRows)
  {
    std::cout << "Only got back " << rowSet.size() << " unique row IDs.\n";
    return;
  }
  //make a new CRS graph based on permuting the rows and columns of mat
  lno_view_t Browmap("rcm perm rowmap", numRows + 1);
  lno_nnz_view_t Bentries("rcm perm entries", nnz);
  //permute rows (compute row counts, then prefix sum, then copy in entries)
  for(lno_t i = 0; i < numRows; i++)
  {
    //row i of B is row perm(i) of A
    Browmap(rcmOrder(i)) = Arowmap(i + 1) - Arowmap(i);
  }
  size_t total = 0;
  for(lno_t i = 0; i <= numRows; i++)
  {
    size_t temp = 0;
    if(i != numRows)
      temp = Browmap(i);
    Browmap(i) = total;
    total += temp;
  }
  for(lno_t i = 0; i < numRows; i++)
  {
    //Copy entries in row i of A into row rcmOrder(i) of B,
    //converting columns too
    for(offset_t j = 0; j < Arowmap(i + 1) - Arowmap(i); j++)
    {
      auto Acol = Aentries(Arowmap(i) + j);
      Bentries(Browmap(rcmOrder(i)) + j) = rcmOrder(Acol);
    }
  }
  //Print sparsity pattern of B
  /*
  std::cout << "A (RCM-reordered):\n\n";
  for(lno_t i = 0; i < numRows; i++)
  {
    std::vector<char> line(numRows, ' ');
    for(offset_t j = Browmap(i); j < Browmap(i + 1); j++)
      line[Bentries(j)] = '*';
    for(lno_t j = 0; j < numRows; j++)
      std::cout << line[j];
    std::cout << '\n';
  }
  std::cout << '\n';
  std::cout << "Change in average bandwidth: " << rcm.find_average_bandwidth(Arowmap, Aentries) << " -> " << rcm.find_average_bandwidth(Browmap, Bentries) << '\n';
  */
}


#define EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE) \
TEST_F( TestCategory, sparse ## _ ## gauss_seidel ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  test_gauss_seidel<SCALAR,ORDINAL,OFFSET,DEVICE>(10000, 10000 * 30, 200, 10); \
} \
TEST_F( TestCategory, sparse ## _ ## rcm ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  test_rcm<SCALAR,ORDINAL,OFFSET,DEVICE>(100, 100, 100); \
} \
TEST_F( TestCategory, sparse ## _ ## cluster_convergence ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  test_cluster_convergence<SCALAR,ORDINAL,OFFSET,DEVICE>("/ascldap/users/bmkelle/msc04515.mtx"); \
} \
TEST_F( TestCategory, sparse ## _ ## rcm_perf ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  test_rcm_perf<SCALAR,ORDINAL,OFFSET,DEVICE>("/ascldap/users/bmkelle/inline_1.mtx"); \
} \
TEST_F( TestCategory, sparse ## _ ## gc_crash ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  test_gc_crash<SCALAR,ORDINAL,OFFSET,DEVICE>("debugGraph.txt"); \
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
 EXECUTE_TEST(kokkos_complex_double, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int64_t, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int64_t, size_t, TestExecSpace)
#endif





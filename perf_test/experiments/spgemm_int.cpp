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
#include <iostream>
#include "KokkosKernels_config.h"
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosKernels_MyCRSMatrix.hpp"
#include "KokkosSparse_spgemm.hpp"
#include "KokkosKernels_TestParameters.hpp"

namespace KokkosKernels{

namespace Experiment{
template <typename crsMat_t, typename device>
bool is_same_matrix(crsMat_t output_mat1, crsMat_t output_mat2){

  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::entries_type::non_const_type   lno_nnz_view_t;
  typedef typename crsMat_t::values_type::non_const_type scalar_view_t;

  size_t nrows1 = output_mat1.graph.row_map.extent(0);
  size_t nentries1 = output_mat1.graph.entries.extent(0) ;
  size_t nvals1 = output_mat1.values.extent(0);

  size_t nrows2 = output_mat2.graph.row_map.extent(0);
  size_t nentries2 = output_mat2.graph.entries.extent(0) ;
  size_t nvals2 = output_mat2.values.extent(0);


  lno_nnz_view_t h_ent1 (Kokkos::ViewAllocateWithoutInitializing("e1"), nentries1);
  scalar_view_t h_vals1 (Kokkos::ViewAllocateWithoutInitializing("v1"), nvals1);


  KokkosKernels::Impl::kk_sort_graph<typename graph_t::row_map_type,
    typename graph_t::entries_type,
    typename crsMat_t::values_type,
    lno_nnz_view_t,
    scalar_view_t,
    typename device::execution_space
    >(
    output_mat1.graph.row_map, output_mat1.graph.entries, output_mat1.values,
    h_ent1, h_vals1
  );

  lno_nnz_view_t h_ent2 (Kokkos::ViewAllocateWithoutInitializing("e1"), nentries2);
  scalar_view_t h_vals2 (Kokkos::ViewAllocateWithoutInitializing("v1"), nvals2);

  if (nrows1 != nrows2) {
	  std::cerr <<"row count is different" << std::endl;
	  return false;
  }
  if (nentries1 != nentries2) {
	  std::cerr <<"nentries2 is different" << std::endl;
	  return false;
  }
  if (nvals1 != nvals2) {
	  std::cerr <<"nvals1 is different" << std::endl;
	  return false;
  }

  KokkosKernels::Impl::kk_sort_graph
      <typename graph_t::row_map_type,
      typename graph_t::entries_type,
      typename crsMat_t::values_type,
      lno_nnz_view_t,
      scalar_view_t,
      typename device::execution_space
      >(
      output_mat2.graph.row_map, output_mat2.graph.entries, output_mat2.values,
      h_ent2, h_vals2
    );

  bool is_identical = true;
  is_identical = KokkosKernels::Impl::kk_is_identical_view
      <typename graph_t::row_map_type, typename graph_t::row_map_type, typename lno_view_t::value_type,
      typename device::execution_space>(output_mat1.graph.row_map, output_mat2.graph.row_map, 0);
  if (!is_identical) {
	  std::cerr << "rowmaps differ" << std::endl;
	  return false;
  }

  is_identical = KokkosKernels::Impl::kk_is_identical_view
      <lno_nnz_view_t, lno_nnz_view_t, typename lno_nnz_view_t::value_type,
      typename device::execution_space>(h_ent1, h_ent2, 0 );
  if (!is_identical) {
	  for (size_t i = 0; i <  nrows1; ++i){
		  size_t rb = output_mat1.graph.row_map[i];
		  size_t re = output_mat1.graph.row_map[i + 1];
		  bool incorrect =false;
		  for (size_t j = rb; j <  re; ++j){
			 if (h_ent1[j] != h_ent2[j]){
				 incorrect = true;
				 break;
			 }
		  }
		  if (incorrect){
			  for (size_t j = rb; j <  re; ++j){
				 	 std::cerr << "row:" << i << " j:" << j <<   " h_ent1[j]:" << h_ent1[j]  << " h_ent2[j]:" << h_ent2[j] << " rb:" << rb << " re:" << re<< std::endl;
			  }
		  }

	  }
	  std::cerr << "entries differ" << std::endl;
	  return false;
  }

  is_identical = KokkosKernels::Impl::kk_is_identical_view
      <scalar_view_t, scalar_view_t, typename scalar_view_t::value_type,
      typename device::execution_space>(h_vals1, h_vals2, 0.000001);
  if (!is_identical) {
    std::cerr << "Incorret values" << std::endl;
  }
  return true;
}


template <typename ExecSpace, typename crsMat_t, typename crsMat_t2 , typename crsMat_t3 , typename TempMemSpace , typename PersistentMemSpace >
crsMat_t3 run_experiment(
    crsMat_t crsMat, crsMat_t2 crsMat2, Parameters params){
    //int algorithm, int repeat, int chunk_size ,int multi_color_scale, int shmemsize, int team_size, int use_dynamic_scheduling, int verbose){

  using namespace KokkosSparse;
  using namespace KokkosSparse::Experimental;
  int algorithm = params.algorithm;
  int repeat = params.repeat;
  int chunk_size = params.chunk_size;

  int shmemsize = params.shmemsize;
  int team_size = params.team_size;
  int use_dynamic_scheduling = params.use_dynamic_scheduling;
  int verbose = params.verbose;
  int calculate_read_write_cost = params.calculate_read_write_cost;
  //char spgemm_step = params.spgemm_step;
  int vector_size = params.vector_size;
  int check_output = params.check_output;
  int mkl_keep_output = params.mkl_keep_output;
  //spgemm_step++;
  typedef typename crsMat_t3::values_type::non_const_type scalar_view_t;
  typedef typename crsMat_t3::StaticCrsGraphType::row_map_type::non_const_type lno_view_t;
  typedef typename crsMat_t3::StaticCrsGraphType::entries_type::non_const_type lno_nnz_view_t;

  lno_view_t row_mapC;
  lno_nnz_view_t entriesC;
  scalar_view_t valuesC;


  typedef typename lno_nnz_view_t::value_type lno_t;
  typedef typename lno_view_t::value_type size_type;
  typedef typename scalar_view_t::value_type scalar_t;


  typedef KokkosKernels::Experimental::KokkosKernelsHandle
      <size_type,lno_t, scalar_t,
      ExecSpace, TempMemSpace,PersistentMemSpace > KernelHandle;

  typedef typename lno_nnz_view_t::value_type idx;
  typedef typename lno_view_t::value_type size_type;

  KernelHandle kh;
  kh.set_team_work_size(chunk_size);
  kh.set_shmem_size(shmemsize);
  kh.set_suggested_team_size(team_size);
  kh.set_suggested_vector_size(vector_size);

  if (use_dynamic_scheduling){
    kh.set_dynamic_scheduling(true);
  }
  if (verbose){
    kh.set_verbose(true);
  }

  const idx m = crsMat.numRows();
  const idx n = crsMat2.numRows();
  const idx k = crsMat2.numCols();

  if (verbose) std::cout << "m:" << m << " n:" << n << " k:" << k << std::endl;
  if (n < crsMat.numCols()){
    std::cerr << "left.numCols():" << crsMat.numCols() << " right.numRows():" << crsMat2.numRows() << std::endl;
    exit(1);
  }

  typename lno_view_t::HostMirror row_mapC_ref;
  typename lno_nnz_view_t::HostMirror entriesC_ref;
  typename scalar_view_t::HostMirror valuesC_ref;
  typename crsMat_t3::HostMirror Ccrsmat_ref;
  if (check_output)
  {
	  if (verbose) std::cout << "Running a reference algorithm" << std::endl;
    row_mapC_ref = typename lno_view_t::HostMirror ("non_const_lnow_row", m + 1);
    entriesC_ref = typename lno_nnz_view_t::HostMirror ("");
    valuesC_ref = typename scalar_view_t::HostMirror ("");
    KernelHandle sequential_kh;
    sequential_kh.set_team_work_size(chunk_size);
    sequential_kh.set_shmem_size(shmemsize);
    sequential_kh.set_suggested_team_size(team_size);
    sequential_kh.create_spgemm_handle(KokkosSparse::SPGEMM_SERIAL);

    if (use_dynamic_scheduling){
      sequential_kh.set_dynamic_scheduling(true);
    }


    spgemm_symbolic (
        &sequential_kh,
        m,
        n,
        k,
        crsMat.graph.row_map,
        crsMat.graph.entries,
        false,
        crsMat2.graph.row_map,
        crsMat2.graph.entries,
        false,
        row_mapC_ref
    );

    ExecSpace().fence();


    size_type c_nnz_size = sequential_kh.get_spgemm_handle()->get_c_nnz();
    if (c_nnz_size){
      entriesC_ref = typename lno_nnz_view_t::HostMirror (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
      valuesC_ref = typename scalar_view_t::HostMirror (Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);
    }

    spgemm_numeric(
        &sequential_kh,
        m,
        n,
        k,
        crsMat.graph.row_map,
        crsMat.graph.entries,
        crsMat.values,
        false,
        crsMat2.graph.row_map,
        crsMat2.graph.entries,
        crsMat2.values,
        false,
        row_mapC_ref,
        entriesC_ref,
        valuesC_ref
    );
    ExecSpace().fence();

    typename crsMat_t3::HostMirror::StaticCrsGraphType static_graph (entriesC_ref, row_mapC_ref);
    typename crsMat_t3::HostMirror Ccrsmat("CrsMatrixC", k, valuesC_ref, static_graph);
    Ccrsmat_ref = Ccrsmat;
  }

  for (int i = 0; i < repeat; ++i){
	  kh.create_spgemm_handle(KokkosSparse::SPGEMMAlgorithm(algorithm));

	  kh.get_spgemm_handle()->mkl_keep_output = mkl_keep_output;
          kh.get_spgemm_handle()->set_mkl_sort_option(params.mkl_sort_option);

	  //if mkl2 input needs to be converted to 1base.
	  kh.get_spgemm_handle()->mkl_convert_to_1base = true;

	  //250000 default. if cache-mode is used on KNL can increase to 1M.
	  kh.get_spgemm_handle()->MaxColDenseAcc = params.MaxColDenseAcc;

	  if (i == 0){
		  kh.get_spgemm_handle()->set_read_write_cost_calc (calculate_read_write_cost);
	  }
	  //do the compression whether in 2 step, or 1 step.
	  kh.get_spgemm_handle()->set_compression_steps(!params.compression2step);
	  //whether to scale the hash more. default is 1, so no scale.
	  kh.get_spgemm_handle()->set_min_hash_size_scale(params.minhashscale);
	  //max occupancy in 1-level LP hashes. LL hashes can be 100%
	  kh.get_spgemm_handle()->set_first_level_hash_cut_off(params.first_level_hash_cut_off);
	  //min reduction on FLOPs to run compression
	  kh.get_spgemm_handle()->set_compression_cut_off(params.compression_cut_off);



	  row_mapC = lno_view_t
			  ("non_const_lnow_row",
					  m + 1);
	  entriesC = lno_nnz_view_t ("entriesC (empty)", 0);
	  valuesC = scalar_view_t ("valuesC (empty)", 0);


	  Kokkos::Impl::Timer timer1;
	  spgemm_symbolic (
			  &kh,
			  m,
			  n,
			  k,
			  crsMat.graph.row_map,
			  crsMat.graph.entries,
			  false,
			  crsMat2.graph.row_map,
			  crsMat2.graph.entries,
			  false,
			  row_mapC
	  );

	  ExecSpace().fence();
	  double symbolic_time = timer1.seconds();

	  Kokkos::Impl::Timer timer3;
	  size_type c_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
	  if (verbose)  std::cout << "C SIZE:" << c_nnz_size << std::endl;
	  if (c_nnz_size){
		  entriesC = lno_nnz_view_t (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
		  valuesC = scalar_view_t (Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);
	  }

	  spgemm_numeric(
			  &kh,
			  m,
			  n,
			  k,
			  crsMat.graph.row_map,
			  crsMat.graph.entries,
			  crsMat.values,
			  false,

			  crsMat2.graph.row_map,
			  crsMat2.graph.entries,
			  crsMat2.values,
			  false,
			  row_mapC,
			  entriesC,
			  valuesC
	  );
	  ExecSpace().fence();
	  double numeric_time = timer3.seconds();

	  std::cout
	  << "mm_time:" << symbolic_time + numeric_time
	  << " symbolic_time:" << symbolic_time
	  << " numeric_time:" << numeric_time << std::endl;
  }
  if (verbose) {
	  std::cout << "row_mapC:" << row_mapC.extent(0) << std::endl;
	  std::cout << "entriesC:" << entriesC.extent(0) << std::endl;
	  std::cout << "valuesC:" << valuesC.extent(0) << std::endl;
	  KokkosKernels::Impl::print_1Dview(valuesC);
	  KokkosKernels::Impl::print_1Dview(entriesC);
	  KokkosKernels::Impl::print_1Dview(row_mapC);
  }


  if (check_output){

	  typename lno_view_t::HostMirror row_mapC_host = Kokkos::create_mirror_view (row_mapC);
	  typename lno_nnz_view_t::HostMirror entriesC_host = Kokkos::create_mirror_view (entriesC);
	  typename scalar_view_t::HostMirror valuesC_host = Kokkos::create_mirror_view (valuesC);

	    Kokkos::deep_copy (row_mapC_host, row_mapC);

	    Kokkos::deep_copy (entriesC_host, entriesC);
	    Kokkos::deep_copy (valuesC_host, valuesC);

	typename crsMat_t3::HostMirror::StaticCrsGraphType static_graph (entriesC_host, row_mapC_host);
	typename crsMat_t3::HostMirror Ccrsmathost("CrsMatrixC", k, valuesC_host, static_graph);

    bool is_identical = is_same_matrix<typename crsMat_t3::HostMirror, typename crsMat_t3::HostMirror::device_type>(Ccrsmat_ref, Ccrsmathost);
    if (!is_identical){
      std::cerr << "Result differs. If values are differing, might be floating point order error." << std::endl;
      exit(1);
    }
  }


  typename crsMat_t3::StaticCrsGraphType static_graph (entriesC, row_mapC);
  crsMat_t3 Ccrsmat("CrsMatrixC", k, valuesC, static_graph);
  return Ccrsmat;

}

}
}

namespace KokkosKernels{

namespace Experiment{

  template <typename size_type, typename lno_t, typename scalar_t,
            typename exec_space, typename hbm_mem_space, typename sbm_mem_space>
  void run_multi_mem_spgemm(Parameters params){

    typedef exec_space myExecSpace;
    typedef Kokkos::Device<exec_space, hbm_mem_space> myFastDevice;
    typedef Kokkos::Device<exec_space, sbm_mem_space> mySlowExecSpace;

    typedef typename MyKokkosSparse::CrsMatrix<scalar_t, lno_t, myFastDevice, void, size_type > fast_crstmat_t;
    //typedef typename fast_crstmat_t::StaticCrsGraphType fast_graph_t;
    //typedef typename fast_crstmat_t::row_map_type::non_const_type fast_row_map_view_t;
    typedef typename fast_crstmat_t::index_type::non_const_type   fast_cols_view_t;
    typedef typename fast_crstmat_t::values_type::non_const_type fast_values_view_t;
    typedef typename fast_crstmat_t::row_map_type::const_type const_fast_row_map_view_t;
    typedef typename fast_crstmat_t::index_type::const_type   const_fast_cols_view_t;
    typedef typename fast_crstmat_t::values_type::const_type const_fast_values_view_t;

    typedef typename MyKokkosSparse::CrsMatrix<scalar_t, lno_t, mySlowExecSpace, void, size_type > slow_crstmat_t;
    //typedef typename slow_crstmat_t::StaticCrsGraphType slow_graph_t;
    //typedef typename slow_crstmat_t::row_map_type::non_const_type slow_row_map_view_t;
    typedef typename slow_crstmat_t::index_type::non_const_type   slow_cols_view_t;
    typedef typename slow_crstmat_t::values_type::non_const_type slow_values_view_t;
    typedef typename slow_crstmat_t::row_map_type::const_type const_slow_row_map_view_t;
    typedef typename slow_crstmat_t::index_type::const_type   const_slow_cols_view_t;
    typedef typename slow_crstmat_t::values_type::const_type const_slow_values_view_t;

    char *a_mat_file = params.a_mtx_bin_file;
    char *b_mat_file = params.b_mtx_bin_file;
    char *c_mat_file = params.c_mtx_bin_file;

    slow_crstmat_t a_slow_crsmat, b_slow_crsmat, c_slow_crsmat;
    fast_crstmat_t a_fast_crsmat, b_fast_crsmat, c_fast_crsmat;

    //read a and b matrices and store them on slow or fast memory.

    if (params.a_mem_space == 1){
      a_fast_crsmat = KokkosKernels::Impl::read_kokkos_crst_matrix<fast_crstmat_t>(a_mat_file);
    }
    else {
      a_slow_crsmat = KokkosKernels::Impl::read_kokkos_crst_matrix<slow_crstmat_t>(a_mat_file);
    }


    if ((b_mat_file == NULL || strcmp(b_mat_file, a_mat_file) == 0) && params.b_mem_space == params.a_mem_space){
      std::cout << "Using A matrix for B as well" << std::endl;
      b_fast_crsmat = a_fast_crsmat;
      b_slow_crsmat = a_slow_crsmat;
    }
    else if (params.b_mem_space == 1){
      if (b_mat_file == NULL) b_mat_file = a_mat_file;
      b_fast_crsmat = KokkosKernels::Impl::read_kokkos_crst_matrix<fast_crstmat_t>(b_mat_file);
    }
    else {
      if (b_mat_file == NULL) b_mat_file = a_mat_file;
      b_slow_crsmat = KokkosKernels::Impl::read_kokkos_crst_matrix<slow_crstmat_t>(b_mat_file);
    }

    if (params.a_mem_space == 1){
      if (params.b_mem_space == 1){
        if (params.c_mem_space == 1){
          if (params.work_mem_space == 1){
            c_fast_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_crstmat_t,fast_crstmat_t,fast_crstmat_t, hbm_mem_space, hbm_mem_space>
                  (a_fast_crsmat, b_fast_crsmat, params);
          }
          else {
            c_fast_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_crstmat_t,fast_crstmat_t,fast_crstmat_t, sbm_mem_space, sbm_mem_space>
                  (a_fast_crsmat, b_fast_crsmat, params);
          }

        }
        else {
          //C is in slow memory.
          if (params.work_mem_space == 1){
            c_slow_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_crstmat_t,fast_crstmat_t,slow_crstmat_t, hbm_mem_space, hbm_mem_space>
                  (a_fast_crsmat, b_fast_crsmat, params);
          }
          else {
            c_slow_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_crstmat_t,fast_crstmat_t,slow_crstmat_t, sbm_mem_space, sbm_mem_space>
                  (a_fast_crsmat, b_fast_crsmat, params);
          }
        }
      }
      else {
        //B is in slow memory
        if (params.c_mem_space == 1){
          if (params.work_mem_space == 1){
            c_fast_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_crstmat_t,slow_crstmat_t,fast_crstmat_t, hbm_mem_space, hbm_mem_space>
                  (a_fast_crsmat, b_slow_crsmat, params);
          }
          else {
            c_fast_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_crstmat_t,slow_crstmat_t,fast_crstmat_t, sbm_mem_space, sbm_mem_space>
                  (a_fast_crsmat, b_slow_crsmat, params);
          }

        }
        else {
          //C is in slow memory.
          if (params.work_mem_space == 1){
            c_slow_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_crstmat_t,slow_crstmat_t,slow_crstmat_t, hbm_mem_space, hbm_mem_space>
                  (a_fast_crsmat, b_slow_crsmat, params);
          }
          else {
            c_slow_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, fast_crstmat_t,slow_crstmat_t,slow_crstmat_t, sbm_mem_space, sbm_mem_space>
                  (a_fast_crsmat, b_slow_crsmat, params);
          }
        }

      }
    }
    else {
      //A is in slow memory
      if (params.b_mem_space == 1){
        if (params.c_mem_space == 1){
          if (params.work_mem_space == 1){
            c_fast_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_crstmat_t,fast_crstmat_t,fast_crstmat_t, hbm_mem_space, hbm_mem_space>
                  (a_slow_crsmat, b_fast_crsmat, params);
          }
          else {
            c_fast_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_crstmat_t,fast_crstmat_t,fast_crstmat_t, sbm_mem_space, sbm_mem_space>
                  (a_slow_crsmat, b_fast_crsmat, params);
          }

        }
        else {
          //C is in slow memory.
          if (params.work_mem_space == 1){
            c_slow_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_crstmat_t,fast_crstmat_t,slow_crstmat_t, hbm_mem_space, hbm_mem_space>
                  (a_slow_crsmat, b_fast_crsmat, params);
          }
          else {
            c_slow_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_crstmat_t,fast_crstmat_t,slow_crstmat_t, sbm_mem_space, sbm_mem_space>
                  (a_slow_crsmat, b_fast_crsmat, params);
          }
        }
      }
      else {
        //B is in slow memory
        if (params.c_mem_space == 1){
          if (params.work_mem_space == 1){
            c_fast_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_crstmat_t,slow_crstmat_t,fast_crstmat_t, hbm_mem_space, hbm_mem_space>
                  (a_slow_crsmat, b_slow_crsmat, params);
          }
          else {
            c_fast_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_crstmat_t,slow_crstmat_t,fast_crstmat_t, sbm_mem_space, sbm_mem_space>
                  (a_slow_crsmat, b_slow_crsmat, params);
          }

        }
        else {
          //C is in slow memory.
          if (params.work_mem_space == 1){
            c_slow_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_crstmat_t,slow_crstmat_t,slow_crstmat_t, hbm_mem_space, hbm_mem_space>
                  (a_slow_crsmat, b_slow_crsmat, params);
          }
          else {
            c_slow_crsmat =
                KokkosKernels::Experiment::run_experiment
                  <myExecSpace, slow_crstmat_t,slow_crstmat_t,slow_crstmat_t, sbm_mem_space, sbm_mem_space>
                  (a_slow_crsmat, b_slow_crsmat, params);
          }
        }

      }

    }


    if (c_mat_file != NULL){
      if (params.c_mem_space == 1){

        fast_cols_view_t sorted_adj("sorted adj", c_fast_crsmat.graph.entries.extent(0));
        fast_values_view_t sorted_vals("sorted vals", c_fast_crsmat.graph.entries.extent(0));

        KokkosKernels::Impl::kk_sort_graph
        <const_fast_row_map_view_t, const_fast_cols_view_t, const_fast_values_view_t, fast_cols_view_t, fast_values_view_t, myExecSpace>(
            c_fast_crsmat.graph.row_map,
            c_fast_crsmat.graph.entries,
            c_fast_crsmat.values, sorted_adj, sorted_vals);

        KokkosKernels::Impl::write_graph_bin(
            (lno_t) (c_fast_crsmat.numRows()),
            (size_type) (c_fast_crsmat.graph.entries.extent(0)),
            c_fast_crsmat.graph.row_map.data(),
            sorted_adj.data(),
            sorted_vals.data(),
            c_mat_file);
      }
      else {
        slow_cols_view_t sorted_adj("sorted adj", c_fast_crsmat.graph.entries.extent(0));
        slow_values_view_t sorted_vals("sorted vals", c_fast_crsmat.graph.entries.extent(0));

        KokkosKernels::Impl::kk_sort_graph<
        const_slow_row_map_view_t, const_slow_cols_view_t, const_slow_values_view_t, slow_cols_view_t, slow_values_view_t, myExecSpace>(
            c_slow_crsmat.graph.row_map,
            c_slow_crsmat.graph.entries,
            c_slow_crsmat.values, sorted_adj, sorted_vals);

        KokkosKernels::Impl::write_graph_bin(
            (lno_t) c_slow_crsmat.numRows(),
            (size_type) c_slow_crsmat.graph.entries.extent(0),
            c_slow_crsmat.graph.row_map.data(),
            sorted_adj.data(),
            sorted_vals.data(),
                    c_mat_file);
      }
    }
  }


}
}



#define SIZE_TYPE int
#define INDEX_TYPE int
#define SCALAR_TYPE int

void print_options(){
  std::cerr << "Options\n" << std::endl;

  std::cerr << "\t[Required] BACKEND: '--threads[numThreads]' | '--openmp [numThreads]' | '--cuda [cudaDeviceIndex]'" << std::endl;

  std::cerr << "\t[Required] INPUT MATRIX: '--amtx [left_hand_side.mtx]' -- for C=AxA" << std::endl;

  std::cerr << "\t[Optional] '--algorithm [DEFAULT=KKDEFAULT=KKSPGEMM|KKMEM|KKDENSE]' --> to choose algorithm. KKMEM is outdated, use KKSPGEMM instead." << std::endl;
  std::cerr << "\t[Optional] --bmtx [righ_hand_side.mtx]' for C= AxB" << std::endl;
  std::cerr << "\t[Optional] OUTPUT MATRICES: '--cmtx [output_matrix.mtx]' --> to write output C=AxB"  << std::endl;
  std::cerr << "\t[Optional] --DENSEACCMAX: on CPUs default algorithm may choose to use dense accumulators. This parameter defaults to 250k, which is max k value to choose dense accumulators. This can be increased with more memory bandwidth." << std::endl;
  std::cerr << "\tThe memory space used for each matrix: '--memspaces [0|1|....15]' --> Bits representing the use of HBM for Work, C, B, and A respectively. For example 12 = 1100, will store work arrays and C on HBM. A and B will be stored DDR. To use this enable multilevel memory in Kokkos, check generate_makefile.sh" << std::endl;
  std::cerr << "\tLoop scheduling: '--dynamic': Use this for dynamic scheduling of the loops. (Better performance most of the time)" << std::endl;
  std::cerr << "\tVerbose Output: '--verbose'" << std::endl;
}


int parse_inputs (KokkosKernels::Experiment::Parameters &params, int argc, char **argv){
  for ( int i = 1 ; i < argc ; ++i ) {
    if ( 0 == strcasecmp( argv[i] , "--threads" ) ) {
      params.use_threads = atoi( argv[++i] );
    }
    else if ( 0 == strcasecmp( argv[i] , "--openmp" ) ) {
      params.use_openmp = atoi( argv[++i] );
    }
    else if ( 0 == strcasecmp( argv[i] , "--cuda" ) ) {
      params.use_cuda = atoi( argv[++i] ) + 1;
    }
    else if ( 0 == strcasecmp( argv[i] , "--repeat" ) ) {
      params.repeat = atoi( argv[++i] );
    }
    else if ( 0 == strcasecmp( argv[i] , "--hashscale" ) ) {
      params.minhashscale = atoi( argv[++i] );
    }
    else if ( 0 == strcasecmp( argv[i] , "--chunksize" ) ) {
      params.chunk_size = atoi( argv[++i] ) ;
    }
    else if ( 0 == strcasecmp( argv[i] , "--teamsize" ) ) {
      params.team_size = atoi( argv[++i] ) ;
    }
    else if ( 0 == strcasecmp( argv[i] , "--vectorsize" ) ) {
      params.vector_size  = atoi( argv[++i] ) ;
    }

    else if ( 0 == strcasecmp( argv[i] , "--compression2step" ) ) {
      params.compression2step =  true ;
    }
    else if ( 0 == strcasecmp( argv[i] , "--shmem" ) ) {
      params.shmemsize =  atoi( argv[++i] ) ;
    }
    else if ( 0 == strcasecmp( argv[i] , "--memspaces" ) ) {
      int memspaces = atoi( argv[++i] ) ;
      int memspaceinfo = memspaces;
      std::cout << "memspaceinfo:" << memspaceinfo << std::endl;
      if (memspaceinfo & 1){
        params.a_mem_space = 1;
        std::cout << "Using HBM for A" << std::endl;
      }
      else {
        params.a_mem_space = 0;
        std::cout << "Using DDR4 for A" << std::endl;
      }
      memspaceinfo  = memspaceinfo >> 1;
      if (memspaceinfo & 1){
        params.b_mem_space = 1;
        std::cout << "Using HBM for B" << std::endl;
      }
      else {
        params.b_mem_space = 0;
        std::cout << "Using DDR4 for B" << std::endl;
      }
      memspaceinfo  = memspaceinfo >> 1;
      if (memspaceinfo & 1){
        params.c_mem_space = 1;
        std::cout << "Using HBM for C" << std::endl;
      }
      else {
        params.c_mem_space = 0;
        std::cout << "Using DDR4 for C" << std::endl;
      }
      memspaceinfo  = memspaceinfo >> 1;
      if (memspaceinfo & 1){
        params.work_mem_space = 1;
        std::cout << "Using HBM for work memory space" << std::endl;
      }
      else {
        params.work_mem_space = 0;
        std::cout << "Using DDR4 for work memory space" << std::endl;
      }
      memspaceinfo  = memspaceinfo >> 1;
    }
    else if ( 0 == strcasecmp( argv[i] , "--CRWC" ) ) {
      params.calculate_read_write_cost = 1;
    }
    else if ( 0 == strcasecmp( argv[i] , "--CIF" ) ) {
      params.coloring_input_file = argv[++i];
    }
    else if ( 0 == strcasecmp( argv[i] , "--COF" ) ) {
      params.coloring_output_file = argv[++i];
    }
    else if ( 0 == strcasecmp( argv[i] , "--CCO" ) ) {
        //if 0.85 set, if compression does not reduce flops by at least 15% symbolic will run on original matrix.
    	//otherwise, it will compress the graph and run symbolic on compressed one.
      params.compression_cut_off = atof( argv[++i] ) ;
    }
    else if ( 0 == strcasecmp( argv[i] , "--FLHCO" ) ) {
    	//if linear probing is used as hash, what is the max occupancy percantage we allow in the hash.
        params.first_level_hash_cut_off = atof( argv[++i] ) ;
    }

    else if ( 0 == strcasecmp( argv[i] , "--flop" ) ) {
    	//print flop statistics. only for the first repeat.
        params.calculate_read_write_cost = 1;
    }

    else if ( 0 == strcasecmp( argv[i] , "--mklsort" ) ) {
    	//when mkl2 is run, the sort option to use.
    	//7:not to sort the output
    	//8:to sort the output
        params.mkl_sort_option = atoi( argv[++i] ) ;
    }
    else if ( 0 == strcasecmp( argv[i] , "--mklkeepout" ) ) {
    	//mkl output is not kept.
        params.mkl_keep_output = atoi( argv[++i] ) ;
    }
    else if ( 0 == strcasecmp( argv[i] , "--checkoutput" ) ) {
    	//check correctness
        params.check_output = 1;
    }
    else if ( 0 == strcasecmp( argv[i] , "--amtx" ) ) {
    	//A at C=AxB
        params.a_mtx_bin_file = argv[++i];
    }

    else if ( 0 == strcasecmp( argv[i] , "--bmtx" ) ) {
    	//B at C=AxB.
    	//if not provided, C = AxA will be performed.
    	params.b_mtx_bin_file = argv[++i];
    }
    else if ( 0 == strcasecmp( argv[i] , "--cmtx" ) ) {
    	//if provided, C will be written to given file.
    	//has to have ".bin", or ".crs" extension.
    	params.c_mtx_bin_file = argv[++i];
    }
    else if ( 0 == strcasecmp( argv[i] , "--dynamic" ) ) {
    	//dynamic scheduling will be used for loops.
    	//currently it is default already.
    	//so has to use the dynamic schedulin.
        params.use_dynamic_scheduling = 1;
    }
    else if ( 0 == strcasecmp( argv[i] , "--DENSEACCMAX" ) ) {
    	//on CPUs and KNLs if DEFAULT algorithm or KKSPGEMM is chosen,
    	//it uses dense accumulators for smaller matrices based on the size of column (k) in B.
    	//Max column size is 250,000 for k to use dense accumulators.
    	//this parameter overwrites this.
    	//with cache mode, or CPUs with smaller thread count, where memory bandwidth is not an issue,
    	//this cut-off can be increased to be more than 250,000
        params.MaxColDenseAcc= atoi( argv[++i] ) ;
    }
    else if ( 0 == strcasecmp( argv[i] , "--verbose" ) ) {
    	//print the timing and information about the inner steps.
    	//if you are timing TPL libraries, for correct timing use verbose option,
    	//because there are pre- post processing in these TPL kernel wraps.
        params.verbose = 1;
    }
    else if ( 0 == strcasecmp( argv[i] , "--algorithm" ) ) {
      ++i;

      if ( 0 == strcasecmp( argv[i] , "DEFAULT" ) ) {
    	  params.algorithm = KokkosSparse::SPGEMM_KK;
      }
      else if ( 0 == strcasecmp( argv[i] , "KKDEFAULT" ) ) {
    	  params.algorithm = KokkosSparse::SPGEMM_KK;
      }
      else if ( 0 == strcasecmp( argv[i] , "KKSPGEMM" ) ) {
    	  params.algorithm = KokkosSparse::SPGEMM_KK;
      }

      else if ( 0 == strcasecmp( argv[i] , "KKMEM" ) ) {
    	  params.algorithm = KokkosSparse::SPGEMM_KK_MEMORY;
      }
      else if ( 0 == strcasecmp( argv[i] , "KKDENSE" ) ) {
        params.algorithm =  KokkosSparse::SPGEMM_KK_DENSE;
      }
      else if ( 0 == strcasecmp( argv[i] , "KKLP" ) ) {
    	  params.algorithm = KokkosSparse::SPGEMM_KK_LP;
      }
      else if ( 0 == strcasecmp( argv[i] , "KKDEBUG" ) ) {
    	  params.algorithm = KokkosSparse::SPGEMM_KK_LP;
      }
      else {
        std::cerr << "Unrecognized command line argument #" << i << ": " << argv[i] << std::endl ;
        print_options();
        return 1;
      }
    }
    else {
      std::cerr << "Unrecognized command line argument #" << i << ": " << argv[i] << std::endl ;
      print_options();
      return 1;
    }
  }
  return 0;
}

int main (int argc, char ** argv){

  KokkosKernels::Experiment::Parameters params;

  if (parse_inputs (params, argc, argv) ){
    return 1;
  }
  if (params.a_mtx_bin_file == NULL){
    std::cerr << "Provide a and b matrix files" << std::endl ;
    print_options();
    return 0;
  }
  if (params.b_mtx_bin_file == NULL){
    std::cout << "B is not provided. Multiplying AxA." << std::endl;
  }

  const int num_threads = params.use_openmp; // Assumption is that use_openmp variable is provided as number of threads
  const int device_id = params.use_cuda - 1;

  Kokkos::initialize( Kokkos::InitArguments( num_threads, -1, device_id ) );
  Kokkos::print_configuration(std::cout);


#if defined( KOKKOS_ENABLE_OPENMP )

  if (params.use_openmp) {
#ifdef KOKKOSKERNELS_INST_MEMSPACE_HBWSPACE
    KokkosKernels::Experiment::run_multi_mem_spgemm
    <SIZE_TYPE, INDEX_TYPE, SCALAR_TYPE, Kokkos::OpenMP, Kokkos::Experimental::HBWSpace, Kokkos::HostSpace>(
        params
        );
#else 
    KokkosKernels::Experiment::run_multi_mem_spgemm
    <SIZE_TYPE, INDEX_TYPE, SCALAR_TYPE, Kokkos::OpenMP, Kokkos::OpenMP::memory_space, Kokkos::OpenMP::memory_space>(
        params
        );
#endif
  }
#endif

#if defined( KOKKOS_ENABLE_CUDA )
  if (params.use_cuda) {
#ifdef KOKKOSKERNELS_INST_MEMSPACE_CUDAHOSTPINNEDSPACE
    KokkosKernels::Experiment::run_multi_mem_spgemm
    <SIZE_TYPE, INDEX_TYPE, SCALAR_TYPE, Kokkos::Cuda, Kokkos::Cuda::memory_space, Kokkos::CudaHostPinnedSpace>(
        params
        );
#else
    KokkosKernels::Experiment::run_multi_mem_spgemm
    <SIZE_TYPE, INDEX_TYPE, SCALAR_TYPE, Kokkos::Cuda, Kokkos::Cuda::memory_space, Kokkos::Cuda::memory_space>(
        params
        );

#endif
  }
#endif

  Kokkos::finalize(); 

  return 0;
}


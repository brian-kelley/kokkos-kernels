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

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <KokkosKernels_Handle.hpp>
#include <KokkosSparse_gauss_seidel.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosKernels_IOUtils.hpp>
#include <KokkosBlas1_fill.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosKernels_config.h>
#include "KokkosKernels_default_types.hpp"
#include <iostream>
#include <vector>
#include <string>

using std::cout;
using std::string;
using namespace KokkosSparse;

static char* getNextArg(int& i, int argc, char** argv)
{
  i++;
  if(i >= argc)
  {
    std::cerr << "Error: expected additional command-line argument!\n";
    exit(1);
  }
  return argv[i];
}

struct GS_Parameters
{
  const char* matrix_path;
  bool graph_symmetric = true;
  GSAlgorithm algo = GS_POINT;
  GSDirection direction = GS_FORWARD;
  //Cluster:
  CGSAlgorithm cgs_algo = CGS_DEFAULT;
  int cluster_size = 10;
  bool compact_scalars = true;
  //Two stage:
  bool classic = false;
};

template<typename device_t>
void runGS(const GS_Parameters& params)
{
  typedef default_scalar scalar_t;
  typedef default_lno_t lno_t;
  typedef default_size_type size_type;
  typedef typename device_t::execution_space exec_space;
  typedef typename device_t::memory_space mem_space;
  typedef KokkosKernels::Experimental::KokkosKernelsHandle<size_type, lno_t, scalar_t, exec_space, mem_space, mem_space> KernelHandle;
  typedef typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device_t, void, size_type> crsmat_t;
  //typedef typename crsmat_t::StaticCrsGraphType graph_t;
  typedef typename crsmat_t::values_type::non_const_type scalar_view_t;
  //typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  //typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  crsmat_t A = KokkosKernels::Impl::read_kokkos_crst_matrix<crsmat_t>(params.matrix_path);
  lno_t nrows = A.numRows();
  lno_t ncols = A.numCols();
  if(nrows != ncols)
  {
    cout << "ERROR: Gauss-Seidel only works for square matrices\n";
    Kokkos::finalize();
    exit(1);
  }
  //size_type nnz = A.nnz();
  KernelHandle kh;
  //use a random RHS - uniformly distributed over (-5, 5)
  scalar_view_t b("b", nrows);
  {
    srand(54321);
    auto bhost = Kokkos::create_mirror_view(b);
    for(lno_t i = 0; i < nrows; i++)
    {
      bhost(i) = 10.0 * rand() / RAND_MAX - 5.0;
    }
    Kokkos::deep_copy(b, bhost);
  }
  double bnorm = KokkosBlas::nrm2(b);
  //initial LHS is 0
  scalar_view_t x("x", nrows);
  //how long symbolic/numeric phases take (the graph reuse case isn't that interesting since numeric doesn't do much)
  Kokkos::Timer timer;
  //cluster size of 1 is standard multicolor GS
  if(params.algo == GS_CLUSTER)
  {
    kh.create_gs_handle(params.cgs_algo, CLUSTER_BALLOON, params.compact_scalars, params.cluster_size);
  }
  else
  {
    kh.create_gs_handle(params.algo);
    if(params.algo == GS_TWOSTAGE)
      kh.set_gs_twostage(!params.classic, nrows);
  }
  timer.reset();
  KokkosSparse::Experimental::gauss_seidel_symbolic
    (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, params.graph_symmetric);
  double symbolicTime = timer.seconds();
  std::cout << "\n*** Symbolic time: " << symbolicTime << '\n';
  timer.reset();
  KokkosSparse::Experimental::gauss_seidel_numeric
    (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, A.values, params.graph_symmetric);
  double numericTime = timer.seconds();
  std::cout << "\n*** Numeric time: " << numericTime << '\n';
  timer.reset();
  //Last two parameters are damping factor (should be 1) and sweeps
  switch(params.direction)
  {
    case GS_SYMMETRIC:
      KokkosSparse::Experimental::symmetric_gauss_seidel_apply
        (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, A.values, x, b, true, true, 1.0, 1);
      break;
    case GS_FORWARD:
      KokkosSparse::Experimental::forward_sweep_gauss_seidel_apply
        (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, A.values, x, b, true, true, 1.0, 1);
      break;
    case GS_BACKWARD:
      KokkosSparse::Experimental::backward_sweep_gauss_seidel_apply
        (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, A.values, x, b, true, true, 1.0, 1);
      break;
  }
  double applyTime = timer.seconds();
  std::cout << "\n*** Apply time: " << applyTime << '\n';
  kh.destroy_gs_handle();
  //Now, compute the 2-norm of residual 
  scalar_view_t res("Ax-b", nrows);
  Kokkos::deep_copy(res, b);
  typedef Kokkos::Details::ArithTraits<scalar_t> KAT;
  scalar_t alpha = KAT::one();
  scalar_t beta = -KAT::one();
  KokkosSparse::spmv<scalar_t, crsmat_t, scalar_view_t, scalar_t, scalar_view_t>
    ("N", alpha, A, x, beta, res);
  double resnorm = KokkosBlas::nrm2(res);
  //note: this still works if the solution diverges
  std::cout << "Relative res norm: " << resnorm / bnorm << '\n';
}

int main(int argc, char** argv)
{
  //Expect two args: matrix name and device flag.
  if(argc < 3)
  {
    cout << "Usage: ./sparse_gs matrix.mtx --algorithm\n\n";
    cout << "device can be \"serial\", \"openmp\", \"cuda\" or \"threads\".\n";
    cout << "If device is not given, the default device for this build is used.\n";
    cout << "Add the --sym-graph flag if the matrix is known to be structurally symmetric.\n";
    cout << "4 main algorithms (required, choose one):\n";
    cout << "  --point\n";
    cout << "  --cluster\n";
    cout << "  --twostage\n";
    cout << "  --classic\n\n";
    cout << "Apply direction (default is forward)\n";
    cout << "  --forward\n";
    cout << "  --backward\n";
    cout << "  --symmetric\n";
    cout << "Options for cluster only:\n";
    cout << "  --cluster-size N (default: 10)\n";
    cout << "  --compact-scalars, --no-compact-scalars (default: compact)\n";
    cout << "  --cgs-apply ALGO\n";
    cout << "     ALGO may be: \"range\", \"team\", \"permuted-range\" or \"permuted-team\".\n";
    cout << "     Default is chosen by the library.\n";
    return 0;
  }
  Kokkos::initialize(argc, argv);
  //device is just the name of the execution space, lowercase
  string device;
  GS_Parameters params;

struct GS_Parameters
{
  std::string matrix_path;
  bool graph_symmetric = true;
  GSAlgorithm algo = GS_POINT;
  GSDirection direction = GS_FORWARD;
  //Cluster:
  CGSAlgorithm cgs_algo = CGS_DEFAULT;
  int cluster_size = 10;
  bool compact_scalars = true;
  //Two stage:
  bool classic = false;
};

  for(int i = 1; i < argc; i++)
  {
    /*
    std::cout << "Usage: ./sparse_gs matrix.mtx --algorithm [--device] [--symmetric]\n\n";
    std::cout << "device can be \"serial\", \"openmp\", \"cuda\" or \"threads\".\n";
    std::cout << "If device is not given, the default device is used.\n";
    std::cout << "Add the --symmetric flag if the matrix is known to be structurally symmetric.\n";
    std::cout << "4 main algorithms (required to pick one):\n";
    std::cout << "  --point\n";
    std::cout << "  --cluster\n";
    std::cout << "  --twostage\n";
    std::cout << "  --classic\n\n";
    std::cout << "Direction flags:
    std::cout << "Options for cluster only:\n";
    std::cout << "  --cluster-size N (default: 10)\n";
    std::cout << "  --compact-scalars, --no-compact-scalars (default: compact)\n";
    std::cout << "  --cgs-apply ALGO\n";
    std::cout << "     ALGO may be: \"range\", \"team\", \"permuted-range\" or \"permuted-team\".\n";
    */
    if(!strcmp(argv[i], "--serial"))
      device = "serial";
    else if(!strcmp(argv[i], "--openmp"))
      device = "openmp";
    else if(!strcmp(argv[i], "--threads"))
      device = "threads";
    else if(!strcmp(argv[i], "--cuda"))
      device = "cuda";
    else if(!strcmp(argv[i], "--sym-graph"))
      params.graph_symmetric  = true;
    else if(!strcmp(argv[i], "--symmetric"))
      params.direction = GS_SYMMETRIC;
    else if(!strcmp(argv[i], "--forward"))
      params.direction = GS_FORWARD;
    else if(!strcmp(argv[i], "--backward"))
      params.direction = GS_BACKWARD;
    else if(!strcmp(argv[i], "--point"))
      params.algo = GS_POINT;
    else if(!strcmp(argv[i], "--cluster"))
      params.algo = GS_CLUSTER;
    else if(!strcmp(argv[i], "--twostage"))
      params.algo = GS_TWOSTAGE;
    else if(!strcmp(argv[i], "--classic"))
    {
      params.algo = GS_TWOSTAGE;
      params.classic = true;
    }
    else if(!strcmp(argv[i], "--compact-scalars"))
      params.compact_scalars = true;
    else if(!strcmp(argv[i], "--no-compact-scalars"))
      params.compact_scalars = false;
    else if(!strcmp(argv[i], "--cgs-apply"))
    {
      const char* cgsApply = getNextArg(i, argc, argv);
      if(!strcmp(cgsApply, "range"))
        params.cgs_algo = CGS_RANGE;
      else if(!strcmp(cgsApply, "team"))
        params.cgs_algo = CGS_TEAM;
      else if(!strcmp(cgsApply, "permuted-range"))
        params.cgs_algo = CGS_PERMUTED_RANGE;
      else if(!strcmp(cgsApply, "permuted-team"))
        params.cgs_algo = CGS_PERMUTED_TEAM;
      else
      {
        std::cout << "\"" << cgsApply << "\" is not a valid cluster GS apply algorithm.\n";
        std::cout << "Valid choices are: range, team, permuted-range, permuted-team.\\n";
        Kokkos::finalize();
        exit(1);
      }
    }
    else if(!strcmp(argv[i], "--cluster-size"))
      params.cluster_size = atoi(getNextArg(i, argc, argv));
    else
      params.matrix_path = argv[i];
  }
  if(!device.length())
  {
    runGS<Kokkos::DefaultExecutionSpace>(params);
  }
  bool run = false;
  #ifdef KOKKOS_ENABLE_SERIAL
  if(device == "serial")
  {
    runGS<Kokkos::Serial>(params);
    run = true;
  }
  #endif
  #ifdef KOKKOS_ENABLE_OPENMP
  if(device == "openmp")
  {
    runGS<Kokkos::OpenMP>(params);
    run = true;
  }
  #endif
  #ifdef KOKKOS_ENABLE_THREADS
  if(device == "threads")
  {
    runGS<Kokkos::Threads>(params);
    run = true;
  }
  #endif
  #ifdef KOKKOS_ENABLE_CUDA
  if(device == "cuda")
  {
    runGS<Kokkos::Cuda>(params);
    run = true;
  }
  #endif
  if(!run)
  {
    std::cerr << "Error: device " << device << " was requested but it's not enabled in this build.\n";
    return 1;
  }
  Kokkos::finalize();
  return 0;
}


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

#include <KokkosKernels_config.h>
#include "KokkosKernels_default_types.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosKernels_Handle.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_gauss_seidel.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosBlas1_dot.hpp"
#include "KokkosBlas1_axpby.hpp"
#include "impl/KokkosSparse_sor_sequential_impl.hpp"

#include <iostream>
#include <string>

using std::string;
using std::cout;
using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;

//Parameters for the Gauss-Seidel preconditioner
struct GS_Parameters
{
  string matrix_path;
  int sweeps = 1; //GS sweeps per CG iteration
  bool graph_symmetric = false;
  //Whether to use any preconditioner
  bool precondition = true;
  //Whether to use sequential GS
  bool sequential = false;
  //Settings for parallel GS
  GSAlgorithm algo = GS_POINT;
  GSDirection direction = GS_FORWARD;
  //Cluster:
  CGSAlgorithm cgs_algo = CGS_DEFAULT;
  CoarseningAlgorithm coarse_algo = CLUSTER_DEFAULT;
  int cluster_size = 10;
  bool compact_scalars = true;
  //Two stage:
  bool classic = false;
};

template <typename device_t>
void runPCG(const GS_Parameters& params)
{
  using scalar_t = default_scalar;
  using lno_t = default_lno_t;
  using size_type = default_size_type;
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using crsMat_t = CrsMatrix<scalar_t, lno_t, device_t, void, size_type>;
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle
    <size_type, lno_t, scalar_t, exec_space, mem_space, mem_space>;
  using scalar_view_t = typename crsMat_t::values_type::non_const_type;
  using vector_t = scalar_view_t;
  using offset_view_t = typename crsMat_t::row_map_type;
  using lno_view_t = typename crsMat_t::index_type;
  using host_offset_view_t = typename offset_view_t::HostMirror;
  using host_lno_view_t = typename lno_view_t::HostMirror;
  using host_scalar_view_t = typename scalar_view_t::HostMirror;
  using KAT = Kokkos::ArithTraits<scalar_t>;

  std::cout << "Loading matrix from " << params.matrix_path << "\n";
  crsMat_t A = KokkosKernels::Impl::read_kokkos_crst_matrix<crsMat_t>(params.matrix_path.c_str());

  lno_t nrows = A.numRows();
  if(A.numCols() != nrows) {
    throw std::invalid_argument("Error: A must be square for PCG to be defined");
  }

  scalar_view_t kok_x_original(Kokkos::ViewAllocateWithoutInitializing("X"), nrows);
  {
    typename scalar_view_t::HostMirror h_x = Kokkos::create_mirror_view(kok_x_original);
    for(lno_t i = 0; i < nrows; ++i)
    {
      scalar_t r = (10.0 * KAT::one() * rand()) / RAND_MAX;
      h_x(i) = r;
    }
    Kokkos::deep_copy (kok_x_original, h_x);
  }
  vector_t y_vector("Y VECTOR", nrows);
  spmv("N", KAT::one(), A, kok_x_original, KAT::zero(), y_vector);

  //create X vector
  vector_t x_vector("kok_x_vector", nrows);

  double solve_time = 0;
  const unsigned cg_iteration_limit = 100000;
  const double   cg_iteration_tolerance = 1e-7 ;

  KernelHandle kh;

  //Host-side views of matrix and inverse diagonal (only initialized if using sequential GS)
  host_offset_view_t ptrHost;
  host_lno_view_t indHost;
  host_scalar_view_t valHost;
  host_scalar_view_t invDiagHost;
  if(params.precondition)
  {
    //Set up preconditioner, depending on the algorithm requested
    if(!params.sequential)
    {
      if(params.cluster_size == 1)
      {
        kh.create_gs_handle(params.algo);
        if(params.algo == GS_TWOSTAGE)
          kh.set_gs_twostage(!params.classic, nrows);
      }
      else
        kh.create_gs_handle(params.cgs_algo, params.coarse_algo, params.compact_scalars, params.cluster_size);
    }
    else
    {
      //Set up for host sequential GS
      ptrHost = Kokkos::create_mirror_view(A.graph.row_map);
      indHost = Kokkos::create_mirror_view(A.graph.entries);
      valHost = Kokkos::create_mirror_view(A.values);
      Kokkos::deep_copy(ptrHost, A.graph.row_map);
      Kokkos::deep_copy(indHost, A.graph.entries);
      Kokkos::deep_copy(valHost, A.values);
      invDiagHost = host_scalar_view_t(Kokkos::ViewAllocateWithoutInitializing("Diag for Seq SOR"), nrows);
      for(int i = 0; i < nrows; i++)
      {
        for(size_type j = ptrHost(i); j < ptrHost(i + 1); j++)
        {
          if(indHost(j) == i)
            invDiagHost(i) = KAT::one() / valHost(j);
        }
      }
    }
  }

  Kokkos::Impl::Timer timer1;

  size_t iteration = 0 ;
  double iter_time = 0 ;
  double matvec_time = 0 ;
  double norm_res = 0 ;
  double precond_time = 0;
  double precond_init_time = 0;

  Kokkos::Impl::Timer wall_clock ;
  Kokkos::Impl::Timer timer;

  // Need input vector to matvec to be owned + received
  scalar_view_t pAll ( "cg::p" , nrows );

  scalar_view_t p = Kokkos::subview( pAll , std::pair<size_t,size_t>(0,nrows) );
  scalar_view_t r ( "cg::r" , nrows );
  scalar_view_t Ap( "cg::Ap", nrows );

  /* r = b - A * x ; */
  /* p  = x       */  Kokkos::deep_copy( p , x_vector );

  /* Ap = A * p   */  spmv("N", 1, A, pAll, 0, Ap);

  /* r  = Ap       */  Kokkos::deep_copy( r , Ap );

  /* r = b - r   */  KokkosBlas::axpby(1.0, y_vector, -1.0, r);

  /* p  = r       */  Kokkos::deep_copy( p , r );

  double old_rdot = KokkosBlas::dot( r , r );
  norm_res  = sqrt( old_rdot );

  scalar_view_t z;

  double precond_old_rdot = 1;
  //Kokkos::deep_copy( p , z );

  bool use_par_sgs = params.precondition && !params.sequential;

  if(params.precondition)
  {
    timer.reset();
    z = scalar_view_t( "pcg::z" , nrows );
    if (use_par_sgs)
    {
      gauss_seidel_symbolic
        (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, params.graph_symmetric);
      gauss_seidel_numeric
        (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, A.values, params.graph_symmetric);

      exec_space().fence();

      precond_init_time += timer.seconds();
      timer.reset();
      //Do initial precondition, that will zero out X and initialize the permuted B, if used
      switch(params.direction)
      {
        case GS_FORWARD:
          forward_sweep_gauss_seidel_apply
            (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, A.values, z, r, true, true, 1.0, params.sweeps);
          break;
        case GS_BACKWARD:
          backward_sweep_gauss_seidel_apply
            (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, A.values, z, r, true, true, 1.0, params.sweeps);
          break;
        case GS_SYMMETRIC:
          symmetric_gauss_seidel_apply
            (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, A.values, z, r, true, true, 1.0, params.sweeps);
      }
      exec_space().fence();
    }
    else if(params.sequential)
    {
      //z = LHS (aka x), r RHS (aka y or b)
      Kokkos::deep_copy(z, KAT::zero());
      auto zhost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z);
      auto rhost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
      //as with par_sgs, init unknown to 0
      timer.reset();
      //then do one initial precondition
      for(int sweep = 0; sweep < params.sweeps; sweep++)
      {
        KokkosSparse::Impl::Sequential::gaussSeidel<lno_t, size_type, scalar_t, scalar_t, scalar_t>
          (nrows, // rows = cols of the matrix
           1,           // number of vectors in X and B
           ptrHost.data(), indHost.data(), valHost.data(),
           rhost.data(), nrows, //raw ptr to B vector, and B column stride (for when multiple RHS gets added to MTSGS)
           zhost.data(), nrows, //raw ptr to X vector, and X column stride
           invDiagHost.data(),
           KAT::one(),
           "F");
        KokkosSparse::Impl::Sequential::gaussSeidel<lno_t, size_type, scalar_t, scalar_t, scalar_t>
          (nrows, 1,
           ptrHost.data(), indHost.data(), valHost.data(),
           rhost.data(), nrows,
           zhost.data(), nrows,
           invDiagHost.data(),
           KAT::one(),
           "B");
      }
      //result is in z (but r doesn't change)
      Kokkos::deep_copy(z, zhost);
      Kokkos::deep_copy(r, rhost);
    }
    precond_time += timer.seconds();
    precond_old_rdot = KokkosBlas::dot(r , z);
    Kokkos::deep_copy(p, z);
  }

  iteration = 0 ;

#ifdef KK_TICTOCPRINT

  std::cout << "norm_res:" << norm_res << " old_rdot:" << old_rdot <<  std::endl;

#endif
  while (cg_iteration_tolerance < norm_res && iteration < cg_iteration_limit) {
    std::cout << "Running CG iteration " << iteration << ", current resnorm = " << norm_res << '\n';

    timer.reset();
    /* Ap = A * p   */  KokkosSparse::spmv("N", 1, A, pAll, 0, Ap);

    exec_space().fence();
    matvec_time += timer.seconds();

    //const double pAp_dot = Kokkos::Example::all_reduce( dot( count_owned , p , Ap ) , import.comm );
    //const double pAp_dot = dot<y_vector_t,y_vector_t, Space>( nrows , p , Ap ) ;

    /* pAp_dot = dot(Ap , p ) */ const double pAp_dot = KokkosBlas::dot( p , Ap ) ;

    double alpha  = 0;
    if (params.precondition){
      alpha = precond_old_rdot / pAp_dot ;
    }
    else {
      alpha = old_rdot / pAp_dot ;
    }

    /* x +=  alpha * p ;  */  KokkosBlas::axpby(alpha, p, 1.0, x_vector);

    /* r += -alpha * Ap ; */  KokkosBlas::axpby(-alpha, Ap, 1.0, r);

    const double r_dot = KokkosBlas::dot( r , r );

    const double beta_original  = r_dot / old_rdot ;
    double precond_r_dot = 1;
    double precond_beta = 1;
    if(params.precondition)
    {
      timer.reset();
      if (use_par_sgs)
      {
        switch(params.direction)
        {
          case GS_FORWARD:
            forward_sweep_gauss_seidel_apply
              (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, A.values, z, r, true, true, 1.0, params.sweeps);
            break;
          case GS_BACKWARD:
            backward_sweep_gauss_seidel_apply
              (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, A.values, z, r, true, true, 1.0, params.sweeps);
            break;
          case GS_SYMMETRIC:
            symmetric_gauss_seidel_apply
              (&kh, nrows, nrows, A.graph.row_map, A.graph.entries, A.values, z, r, true, true, 1.0, params.sweeps);
        }
      }
      else if(params.sequential)
      {
        //z = LHS (aka x), r RHS (aka y or b)
        Kokkos::deep_copy(z, 0.0);
        auto zhost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z);
        auto rhost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
        //as with the par_sgs version, init unknown (here, zhost) to 0
        for(int sweep = 0; sweep < params.sweeps; sweep++)
        {
          if(params.direction == GS_FORWARD
              || params.direction == GS_SYMMETRIC)
          {
            KokkosSparse::Impl::Sequential::gaussSeidel<lno_t, size_type, scalar_t, scalar_t, scalar_t>
              (nrows, 1,
               ptrHost.data(), indHost.data(), valHost.data(),
               rhost.data(), nrows,
               zhost.data(), nrows,
               invDiagHost.data(),
               KAT::one(),
               "F");
          }
          if(params.direction == GS_BACKWARD
              || params.direction == GS_SYMMETRIC)
          {
            KokkosSparse::Impl::Sequential::gaussSeidel<lno_t, size_type, scalar_t, scalar_t, scalar_t>
              (nrows, nrows,
               ptrHost.data(), indHost.data(), valHost.data(),
               rhost.data(), nrows,
               zhost.data(), nrows,
               invDiagHost.data(),
               KAT::one(),
               "B");
          }
        }
        Kokkos::deep_copy(z, zhost);
        Kokkos::deep_copy(r, rhost);
      }
      precond_time += timer.seconds();
      precond_r_dot = KokkosBlas::dot(r , z );
      precond_beta  = precond_r_dot / precond_old_rdot ;
    }
    scalar_t beta;
    if (!params.precondition) {
      beta = beta_original;
      /* p = r + beta * p ; */  KokkosBlas::axpby(1.0, r, beta, p);
    }
    else {
      beta = precond_beta;
      KokkosBlas::axpby(1.0, z, beta, p);
    }

#ifdef KK_TICTOCPRINT
    std::cout << "\tbeta_original:" << beta_original <<  std::endl;
    if (use_sgs)
      std::cout << "\tprecond_beta:" << precond_beta <<  std::endl;

#endif

    norm_res = sqrt( old_rdot = r_dot );
    precond_old_rdot = precond_r_dot;

#ifdef KK_TICTOCPRINT
    std::cout << "\tnorm_res:" << norm_res << " old_rdot:" << old_rdot<<  std::endl;
#endif
    ++iteration ;
  }

  exec_space().fence();
  iter_time = wall_clock.seconds();

  if(params.precondition && !params.sequential) {
    kh.destroy_gs_handle();
  }

  solve_time = timer1.seconds();

  string algoSummary;
  if(params.precondition) {
    if(params.sequential)
      algoSummary = "SEQUENTIAL GS";
    else
    {
      if(params.cluster_size == 1)
      {
        if(params.algo == GS_TWOSTAGE)
          algoSummary = "TWO-STAGE/CLASSIC GS";
        else
          algoSummary = "POINT-COLORING GS";
      }
      else
        algoSummary = "CLUSTER-COLORING GS (CLUSTER SIZE " + std::to_string(params.cluster_size) + ")";
    }
  }
  else {
    algoSummary = "NO";
  }

  cout  << "DEFAULT SOLVE: " << algoSummary << " PRECONDITIONER"
      << "\n\t(P)CG_NUM_ITER              [" << iteration << "]"
      << "\n\tMATVEC_TIME                 [" << matvec_time << "]"
      << "\n\tCG_RESIDUAL                 [" << norm_res << "]"
      << "\n\tCG_ITERATION_TIME           [" << iter_time << "]"
      << "\n\tPRECONDITIONER_TIME         [" << precond_time << "]"
      << "\n\tPRECONDITIONER_INIT_TIME    [" << precond_init_time << "]"
      << "\n\tPRECOND_APPLY_TIME_PER_ITER [" << precond_time / (iteration + 1) << "]"
      << "\n\tSOLVE_TIME                  [" << solve_time<< "]\n";
}

static char* getNextArg(int& i, int argc, char** argv)
{
  if(i >= argc)
  {
    std::cerr << "Error: expected additional command-line argument!\n";
    exit(1);
  }
  return argv[i++];
}

int main(int argc, char** argv)
{
  //Expect two args: matrix name and device flag.
  if(argc == 1 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))
  {
    cout << "Usage: ./sparse_gs [--threads=N] [--device-id=K] matrix.mtx [--device-type] [other args]\n\n";
    cout << "\"--device-type\" flag can be \"--serial\", \"--openmp\", \"--cuda\" or \"--threads\".\n";
    cout << "If device is not given, the default device for this build is used.\n";
    cout << "\nOther flags:\n";
    cout << "--sym-graph: promise the matrix is structurally symmetric (OK if symmetric but flag isn't provided)\n";
    cout << "--sweeps S: run preconditioner S times between CG iters. For SGS, forward+backward counts as 1 sweep.\n";
    cout << "5 main choices of preconditioner (default is point):\n";
    cout << "  --point\n";
    cout << "  --cluster\n";
    cout << "  --twostage\n";
    cout << "  --classic\n\n";
    cout << "  --sequential (always runs on host)\n\n";
    cout << "  --no-prec\n";
    cout << "Apply direction (default is forward)\n";
    cout << "  --forward\n";
    cout << "  --backward\n";
    cout << "  --symmetric\n";
    cout << "Options for cluster:\n";
    cout << "  --cluster-size N (default: 10)\n";
    cout << "  --compact-scalars, --no-compact-scalars (default: compact)\n";
    cout << "  --cgs-apply ALGO\n";
    cout << "     ALGO may be: \"range\", \"team\", \"permuted-range\" or \"permuted-team\".\n";
    cout << "     Default is chosen by the library.\n";
    cout << "  --coarse-algo ALGO\n";
    cout << "     ALGO may be: \"mis2\", \"balloon\"\n";
    return 0;
  }
  Kokkos::initialize(argc, argv);
  //device is just the name of the execution space, lowercase
  string deviceName;
  GS_Parameters params;
  int i = 1;
  params.matrix_path = getNextArg(i, argc, argv);
  for(; i < argc; i++)
  {
    if(!strcmp(argv[i], "--serial"))
      deviceName = "serial";
    else if(!strcmp(argv[i], "--openmp"))
      deviceName = "openmp";
    else if(!strcmp(argv[i], "--threads"))
      deviceName = "threads";
    else if(!strcmp(argv[i], "--cuda"))
      deviceName = "cuda";
    else if(!strcmp(argv[i], "--sym-graph"))
      params.graph_symmetric = true;
    else if(!strcmp(argv[i], "--sweeps"))
      params.sweeps = atoi(getNextArg(++i, argc, argv));
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
    else if(!strcmp(argv[i], "--no-prec"))
      params.precondition = false;
    else if(!strcmp(argv[i], "--compact-scalars"))
      params.compact_scalars = true;
    else if(!strcmp(argv[i], "--no-compact-scalars"))
      params.compact_scalars = false;
    else if(!strcmp(argv[i], "--cgs-apply"))
    {
      const char* cgsApply = getNextArg(++i, argc, argv);
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
    else if(!strcmp(argv[i], "--coarse-algo"))
    {
      const char* algo = getNextArg(++i, argc, argv);
      if(!strcmp(algo, "balloon"))
        params.coarse_algo = CLUSTER_BALLOON;
      else if(!strcmp(algo, "mis2"))
        params.coarse_algo = CLUSTER_MIS2;
      else
      {
        std::cout << "Error: invalid coarsening algorithm. Options are balloon and mis2.\n";
        Kokkos::finalize();
        exit(1);
      }
    }
    else if(!strcmp(argv[i], "--cluster-size"))
      params.cluster_size = atoi(getNextArg(++i, argc, argv));
    else
    {
      std::cout << "Unknown argument \"" << argv[i] << "\"\n";
      Kokkos::finalize();
      exit(1);
    }
  }
  bool run = false;
  if(!deviceName.length())
  {
    runPCG<Kokkos::DefaultExecutionSpace>(params);
    run = true;
  }
  #ifdef KOKKOS_ENABLE_SERIAL
  if(deviceName == "serial")
  {
    runPCG<Kokkos::Serial>(params);
    run = true;
  }
  #endif
  #ifdef KOKKOS_ENABLE_OPENMP
  if(deviceName == "openmp")
  {
    runPCG<Kokkos::OpenMP>(params);
    run = true;
  }
  #endif
  #ifdef KOKKOS_ENABLE_THREADS
  if(deviceName == "threads")
  {
    runPCG<Kokkos::Threads>(params);
    run = true;
  }
  #endif
  #ifdef KOKKOS_ENABLE_CUDA
  if(deviceName == "cuda")
  {
    runPCG<Kokkos::Cuda>(params);
    run = true;
  }
  #endif
  if(!run)
  {
    std::cerr << "Error: device " << deviceName << " was requested but it's not enabled in this build.\n";
    return 1;
  }
  Kokkos::finalize();
  return 0;
}


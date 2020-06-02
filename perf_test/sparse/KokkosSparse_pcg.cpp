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
#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_Handle.hpp"
#include "KokkosSparse_gauss_seidel.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosKernels_default_types.hpp"
#include "impl/KokkosSparse_sor_sequential_impl.hpp"
#include "KokkosKernels_IOUtils.hpp"

#include <iostream>
#include <string>

using std::string;
using std::cout;

struct CGSolveResult
{
  size_t iteration ;
  double iter_time ;
  double matvec_time ;
  double norm_res ;
  double precond_time;
  double precond_init_time;
};

//Parameters for the Gauss-Seidel preconditioner
struct GS_Parameters
{
  int sweeps = 1; //GS sweeps per CG iteration
  bool graph_symmetric = true;
  //Whether to use any preconditioner
  bool precondition = true;
  //Whether to use sequential GS
  bool sequential = false;
  //Settings for parallel GS
  GSAlgorithm algo = GS_POINT;
  GSDirection direction = GS_FORWARD;
  //Cluster:
  CGSAlgorithm cgs_algo = CGS_DEFAULT;
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
  using crsMat_t = KokkosSparse::CrsMatrix<scalar_t, lno_t, device_t, void, size_type>;
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle
    <size_type, lno_t, scalar_t, exec_space, mem_space, mem_space>;
  using scalar_view_t = typename crsMat_t::values_type::non_const_type;
  using offset_view_t = typename crsMat_t::StaticCrsGraphType::row_map_type::non_const_type;
  using lno_view_t = typename crsMat_t::StaticCrsGraphType::entries_type::non_const_type;
  using KAT = Kokkos::ArithTraits<scalar_t>;

  crsmat_t A = KokkosKernels::Impl::read_kokkos_crst_matrix<crsMat_t>(params.matrix_path);

  lno_t nrows = A.numRows();
  lno_t ncols = A.numCols();

  scalar_view_t kok_x_original(Kokkos::ViewAllocateWithoutInitializing("X"), ncols);
  {
    typename scalar_view_t::HostMirror h_x = Kokkos::create_mirror_view(kok_x_original);
    for(lno_t i = 0; i < nv; ++i)
    {
      scalar_t r = (max_value * rand()) / RAND_MAX;
      h_x(i) = r;
    }
    Kokkos::deep_copy (kok_x_original, h_x);
  }
  vector_t y_vector("Y VECTOR", nrows);
  KokkosSparse::spmv("N", 1, crsMat, kok_x_original, 1, y_vector);

  //create X vector
  scalar_view_t x_vector("kok_x_vector", ncols);

  double solve_time = 0;
  const unsigned cg_iteration_limit = 100000;
  const double   cg_iteration_tolerance = 1e-7 ;

  KokkosKernels::Experimental::Example::CGSolveResult cg_result;

  KernelHandle kh;

  //Host-side views of matrix and inverse diagonal (only initialized if using sequential GS)
  typename offset_view_t::HostMirror ptrHost;
  typename lno_view_t::HostMirror indHost;
  typename scalar_view_t::HostMirror valHost;
  typename scalar_view_t::HostMirror invDiagHost;

  if(params.precondition)
  {
    //Set up preconditioner, depending on the algorithm requested
    if(!params.sequential)
    {
      if(clusterSize == 1)
      {
        kh.create_gs_handle(params.algo);
        if(params.algo == GS_TWOSTAGE)
          kh.set_gs_twostage(!params.classic, nrows);
      }
      else
        kh.create_gs_handle(params.cgs_algo, CLUSTER_BALLOON, params.compact_scalars, params.cluster_size);
    }
    else
    {
      //Set up for host sequential GS
      ptrHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), crsMat.graph.row_map);
      indHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), crsMat.graph.entries);
      valHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), crsMat.values);
      Kokkos::View<double*, Kokkos::HostSpace> invDiagHost;
      if(use_sequential_sgs)
      {
        diagHost = Kokkos::View<double*, Kokkos::HostSpace>("Diag for Seq SOR", nrows);
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
  scalar_view_t pAll ( "cg::p" , ncols );

  scalar_view_t p = Kokkos::subview( pAll , std::pair<size_t,size_t>(0,nrows) );
  scalar_view_t r ( "cg::r" , nrows );
  scalar_view_t Ap( "cg::Ap", nrows );

  /* r = b - A * x ; */
  /* p  = x       */  Kokkos::deep_copy( p , x_vector );

  /* Ap = A * p   */  KokkosSparse::spmv("N", 1, crsMat, pAll, 0, Ap);

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
    z = scalar_view_t( "pcg::z" , count_total );
    if (use_par_sgs)
    {
      gauss_seidel_symbolic
        (&kh, nr, nc, crsMat.graph.row_map, crsMat.graph.entries, crsMat.values);
      gauss_seidel_numeric
        (&kh, count_total, count_total, crsMat.graph.row_map, crsMat.graph.entries, crsMat.values);

      Space().fence();

      precond_init_time += timer.seconds();
      Space().fence();
      timer.reset();
      //Do initial precondition, that will zero out X and initialize the permuted B, if used
      switch(params.direction)
      {
        case GS_FORWARD:
          forward_sweep_gauss_seidel_apply
            (&kh, count_total, count_total, crsMat.graph.row_map, crsMat.graph.entries, crsMat.values, z, r, true, true, 1.0, params.sweeps);
          break;
        case GS_BACKWARD:
          backward_sweep_gauss_seidel_apply
            (&kh, count_total, count_total, crsMat.graph.row_map, crsMat.graph.entries, crsMat.values, z, r, true, true, 1.0, params.sweeps);
          break;
        case GS_SYMMETRIC:
          symmetric_gauss_seidel_apply
            (&kh, count_total, count_total, crsMat.graph.row_map, crsMat.graph.entries, crsMat.values, z, r, true, true, 1.0, params.sweeps);
      }
      Space().fence();
    }
    else if(use_sequential_sgs)
    {
      //z = LHS (aka x), r RHS (aka y or b)
      Kokkos::deep_copy(z, 0.0);
      auto zhost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z);
      auto rhost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
      //as with par_sgs, init unknown to 0
      timer.reset();
      //then do one initial precondition
      for(int sweep = 0; sweep < apply_count; sweep++)
      {
        KokkosSparse::Impl::Sequential::gaussSeidel<nnz_lno_t, size_type, double, double, double>
          (count_total, // rows = cols of the matrix
           1,           // number of vectors in X and B
           ptrHost.data(), indHost.data(), valHost.data(),
           rhost.data(), count_total, //raw ptr to B vector, and B column stride (for when multiple RHS gets added to MTSGS)
           zhost.data(), count_total, //raw ptr to X vector, and X column stride
           diagHost.data(),
           1.0,
           "F");
        KokkosSparse::Impl::Sequential::gaussSeidel<nnz_lno_t, size_type, double, double, double>
          (count_total, 1,
           ptrHost.data(), indHost.data(), valHost.data(), 
           rhost.data(), count_total,
           zhost.data(), count_total,
           diagHost.data(),
           1.0,
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
    /* Ap = A * p   */  KokkosSparse::spmv("N", 1, crsMat, pAll, 0, Ap);

    Space().fence();
    matvec_time += timer.seconds();

    //const double pAp_dot = Kokkos::Example::all_reduce( dot( count_owned , p , Ap ) , import.comm );
    //const double pAp_dot = dot<y_vector_t,y_vector_t, Space>( count_total , p , Ap ) ;

    /* pAp_dot = dot(Ap , p ) */ const double pAp_dot = KokkosBlas::dot( p , Ap ) ;

    double alpha  = 0;
    if (use_sgs){
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
      Space().fence();
      timer.reset();
      if (use_par_sgs)
      {
        switch(params.direction)
        {
          case GS_FORWARD:
            forward_sweep_gauss_seidel_apply
              (&kh, count_total, count_total, crsMat.graph.row_map, crsMat.graph.entries, crsMat.values, z, r, true, true, 1.0, params.sweeps);
            break;
          case GS_BACKWARD:
            backward_sweep_gauss_seidel_apply
              (&kh, count_total, count_total, crsMat.graph.row_map, crsMat.graph.entries, crsMat.values, z, r, true, true, 1.0, params.sweeps);
            break;
          case GS_SYMMETRIC:
            symmetric_gauss_seidel_apply
              (&kh, count_total, count_total, crsMat.graph.row_map, crsMat.graph.entries, crsMat.values, z, r, true, true, 1.0, params.sweeps);
        }
      }
      else if(use_sequential_sgs)
      {
        //z = LHS (aka x), r RHS (aka y or b)
        Kokkos::deep_copy(z, 0.0);
        auto zhost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z);
        auto rhost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
        //as with the par_sgs version, init unknown (here, zhost) to 0
        for(int sweep = 0; sweep < apply_count; sweep++)
        {
          KokkosSparse::Impl::Sequential::gaussSeidel<nnz_lno_t, size_type, double, double, double>
            (count_total, 1,
             ptrHost.data(), indHost.data(), valHost.data(), 
             rhost.data(), count_total,
             zhost.data(), count_total,
             diagHost.data(),
             1.0,
             "F");
          KokkosSparse::Impl::Sequential::gaussSeidel<nnz_lno_t , size_type, double, double, double>
            (count_total, 1,
             ptrHost.data(), indHost.data(), valHost.data(), 
             rhost.data(), count_total,
             zhost.data(), count_total,
             diagHost.data(),
             1.0,
             "B");
        }
        Kokkos::deep_copy(z, zhost);
        Kokkos::deep_copy(r, rhost);
      }
      precond_time += timer.seconds();
      precond_r_dot = KokkosBlas::dot(r , z );
      precond_beta  = precond_r_dot / precond_old_rdot ;
    }
    double beta = 1;
    if (!use_sgs) {
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

  Space().fence();
  iter_time = wall_clock.seconds();

  result.iteration         = iteration;
  result.iter_time         = iter_time;
  result.matvec_time       = matvec_time;
  result.norm_res          = norm_res;
  result.precond_time      = precond_time;
  result.precond_init_time = precond_init_time;



  Kokkos::fence();

  solve_time = timer1.seconds();

  std::string algoSummary;
  if(useSequential)
    algoSummary = "SEQUENTIAL GS";
  else
  {
    if(clusterSize == 1)
    {
      if(params.algo == GS_TWOSTAGE)
        algoSummary = "TWO-STAGE/CLASSIC GS";
      else
        algoSummary = "POINT-COLORING GS";
    }
    else
      algoSummary = "CLUSTER-COLORING GS (CLUSTER SIZE " + std::to_string(clusterSize) + ")";
  }

  std::cout  << "DEFAULT SOLVE: " << algoSummary << " PRECONDITIONER"
      << "\n\t(P)CG_NUM_ITER              [" << cg_result.iteration << "]"
      << "\n\tMATVEC_TIME                 [" << cg_result.matvec_time << "]"
      << "\n\tCG_RESIDUAL                 [" << cg_result.norm_res << "]"
      << "\n\tCG_ITERATION_TIME           [" << cg_result.iter_time << "]"
      << "\n\tPRECONDITIONER_TIME         [" << cg_result.precond_time << "]"
      << "\n\tPRECONDITIONER_INIT_TIME    [" << cg_result.precond_init_time << "]"
      << "\n\tPRECOND_APPLY_TIME_PER_ITER [" << cg_result.precond_time / (cg_result.iteration  + 1) << "]"
      << "\n\tSOLVE_TIME                  [" << solve_time<< "]"
      << std::endl ;



  /*
  kh.destroy_gs_handle();
  kh.create_gs_handle(KokkosKernels::Experimental::Graph::GS_PERMUTED);

  kok_x_vector = scalar_view_t("kok_x_vector", nv);
  timer1.reset();
  KokkosKernels::Experimental::Example::pcgsolve(
        kh
      ,A 
      , kok_b_vector
      , kok_x_vector
      , cg_iteration_limit
      , cg_iteration_tolerance
      , & cg_result
      , true
  );

  Kokkos::fence();
  solve_time = timer1.seconds();
  std::cout  << "\nPERMUTED SGS SOLVE:"
      << "\n\t(P)CG_NUM_ITER              [" << cg_result.iteration << "]"
      << "\n\tMATVEC_TIME                 [" << cg_result.matvec_time << "]"
      << "\n\tCG_RESIDUAL                 [" << cg_result.norm_res << "]"
      << "\n\tCG_ITERATION_TIME           [" << cg_result.iter_time << "]"
      << "\n\tPRECONDITIONER_TIME         [" << cg_result.precond_time << "]"
      << "\n\tPRECONDITIONER_INIT_TIME    [" << cg_result.precond_init_time << "]"
      << "\n\tPRECOND_APPLY_TIME_PER_ITER [" << cg_result.precond_time / (cg_result.iteration  + 1) << "]"
      << "\n\tSOLVE_TIME                  [" << solve_time<< "]"
      << std::endl ;


  kh.destroy_gs_handle();
  kh.create_gs_handle(KokkosKernels::Experimental::Graph::GS_TEAM);

  kok_x_vector = scalar_view_t("kok_x_vector", nv);
  timer1.reset();
  KokkosKernels::Experimental::Example::pcgsolve(
        kh
      , A
      , kok_b_vector
      , kok_x_vector
      , cg_iteration_limit
      , cg_iteration_tolerance
      , & cg_result
      , true
  );
  Kokkos::fence();

  solve_time = timer1.seconds();
  std::cout  << "\nTEAM SGS SOLVE:"
      << "\n\t(P)CG_NUM_ITER              [" << cg_result.iteration << "]"
      << "\n\tMATVEC_TIME                 [" << cg_result.matvec_time << "]"
      << "\n\tCG_RESIDUAL                 [" << cg_result.norm_res << "]"
      << "\n\tCG_ITERATION_TIME           [" << cg_result.iter_time << "]"
      << "\n\tPRECONDITIONER_TIME         [" << cg_result.precond_time << "]"
      << "\n\tPRECONDITIONER_INIT_TIME    [" << cg_result.precond_init_time << "]"
      << "\n\tPRECOND_APPLY_TIME_PER_ITER [" << cg_result.precond_time / (cg_result.iteration  + 1) << "]"
      << "\n\tSOLVE_TIME                  [" << solve_time<< "]"
      << std::endl ;




  kok_x_vector = scalar_view_t("kok_x_vector", nv);
  timer1.reset();
  KokkosKernels::Experimental::Example::pcgsolve(
        kh
      , A
      , kok_b_vector
      , kok_x_vector
      , cg_iteration_limit
      , cg_iteration_tolerance
      , & cg_result
      , false
  );
  Kokkos::fence();

  solve_time = timer1.seconds();
  std::cout  << "\nCG SOLVE (With no Preconditioner):"
      << "\n\t(P)CG_NUM_ITER              [" << cg_result.iteration << "]"
      << "\n\tMATVEC_TIME                 [" << cg_result.matvec_time << "]"
      << "\n\tCG_RESIDUAL                 [" << cg_result.norm_res << "]"
      << "\n\tCG_ITERATION_TIME           [" << cg_result.iter_time << "]"
      << "\n\tPRECONDITIONER_TIME         [" << cg_result.precond_time << "]"
      << "\n\tPRECONDITIONER_INIT_TIME    [" << cg_result.precond_init_time << "]"
      << "\n\tPRECOND_APPLY_TIME_PER_ITER [" << cg_result.precond_time / (cg_result.iteration  + 1) << "]"
      << "\n\tSOLVE_TIME                  [" << solve_time<< "]"
      << std::endl ;
  */
}

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
      params.sweeps = atoi(getNextArg(i, argc, argv));
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
  bool run = false;
  if(!device.length())
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

/*
//BMK TODO: need some extra work to support block
template< typename KernelHandle_t,
          typename crsMatrix_t,
          typename y_vector_t,
          typename x_vector_t
          >
void block_pcgsolve(
               KernelHandle_t &kh
            ,  const crsMatrix_t &point_crsMat
            ,  const crsMatrix_t &_block_crsMat, int block_size
            ,  const y_vector_t &y_vector
            ,  x_vector_t x_vector
            ,  const size_t  maximum_iteration = 200
            ,  const double  tolerance = std::numeric_limits<double>::epsilon()
            ,  CGSolveResult * result = 0
            ,  bool use_sgs = true)
{
  using namespace KokkosSparse;
  using namespace KokkosSparse::Experimental;
  typedef typename KernelHandle_t::HandleExecSpace Space;

  const size_t count_total = point_crsMat.numRows();

  size_t  iteration = 0 ;
  double  iter_time = 0 ;
  double  matvec_time = 0 ;
  double  norm_res = 0 ;
  double precond_time = 0;
  double precond_init_time = 0;

  Kokkos::Impl::Timer wall_clock ;
  Kokkos::Impl::Timer timer;

  // Need input vector to matvec to be owned + received
  y_vector_t pAll ( "cg::p" , count_total );

  y_vector_t p = Kokkos::subview( pAll , std::pair<size_t,size_t>(0,count_total) );
  y_vector_t r ( "cg::r" , count_total );
  y_vector_t Ap( "cg::Ap", count_total );

  // r = b - A * x ;
  // p  = x      
  Kokkos::deep_copy( p , x_vector );

  // Ap = A * p 
  KokkosSparse::spmv("N", 1, point_crsMat, pAll, 0, Ap);

  // r  = Ap
  Kokkos::deep_copy( r , Ap );

  // r = b - r
  KokkosBlas::axpby(1.0, y_vector, -1.0, r);

  // p  = r
  Kokkos::deep_copy( p , r );
;
  double old_rdot = KokkosBlas::dot( r , r );
  norm_res  = sqrt( old_rdot );

  int apply_count = 1;
  y_vector_t z;

  double precond_old_rdot = 1;
  //Kokkos::deep_copy( p , z );

  bool owner_handle = false;

  KernelHandle_t block_kh;
  block_kh.create_gs_handle();
  block_kh.get_point_gs_handle()->set_block_size(block_size);
    //block_kh.set_shmem_size(8032);
  if (use_sgs){
    if (kh.get_gs_handle() == NULL){
      owner_handle = true;
      kh.create_gs_handle();
    }

    timer.reset();

    //gauss_seidel_numeric
    //  (&kh, count_total, count_total, point_crsMat.graph.row_map, point_crsMat.graph.entries, point_crsMat.values);

    //Space().fence();
    //timer.reset();

    //block_kh.set_verbose(true);
    block_gauss_seidel_numeric
          (&block_kh, _block_crsMat.numRows(), _block_crsMat.numCols(), block_size, _block_crsMat.graph.row_map, _block_crsMat.graph.entries, _block_crsMat.values);

    precond_init_time += timer.seconds();

    z = y_vector_t( "pcg::z" , count_total );
    Space().fence();
    timer.reset();
    symmetric_block_gauss_seidel_apply
            (&block_kh, _block_crsMat.numRows(), _block_crsMat.numCols(),block_size,  _block_crsMat.graph.row_map, _block_crsMat.graph.entries, _block_crsMat.values,
            		z, r, true, true, 1.0, apply_count);

    //symmetric_gauss_seidel_apply
    //    (&kh, count_total, count_total, point_crsMat.graph.row_map, point_crsMat.graph.entries, point_crsMat.values, z, r, true, true, apply_count);
    Space().fence();
    precond_time += timer.seconds();
    precond_old_rdot = KokkosBlas::dot( r , z );
    Kokkos::deep_copy( p , z );
  }

  iteration = 0 ;

#ifdef KK_TICTOCPRINT

  std::cout << "norm_res:" << norm_res << " old_rdot:" << old_rdot<<  std::endl;

#endif
  while ( tolerance < norm_res && iteration < maximum_iteration ) {


    timer.reset();
    //Ap = A * p
    KokkosSparse::spmv("N", 1, point_crsMat, pAll, 0, Ap);


    Space().fence();
    matvec_time += timer.seconds();

    //const double pAp_dot = Kokkos::Example::all_reduce( dot( count_owned , p , Ap ) , import.comm );
    //const double pAp_dot = dot<y_vector_t,y_vector_t, Space>( count_total , p , Ap ) ;

    // pAp_dot = dot(Ap , p);
    const double pAp_dot = KokkosBlas::dot( p , Ap ) ;


    double alpha  = 0;
    if (use_sgs){
      alpha = precond_old_rdot / pAp_dot ;
    }
    else {
      alpha = old_rdot / pAp_dot ;
    }

    // x +=  alpha * p ;
    KokkosBlas::axpby(alpha, p, 1.0, x_vector);

    // r += -alpha * Ap ;
    KokkosBlas::axpby(-alpha, Ap, 1.0, r);

    const double r_dot = KokkosBlas::dot( r , r );

    const double beta_original  = r_dot / old_rdot ;
    double precond_r_dot = 1;
    double precond_beta = 1;
    if (use_sgs){
      Space().fence();
      timer.reset();
      symmetric_block_gauss_seidel_apply
                  (&block_kh, _block_crsMat.numRows(), _block_crsMat.numCols(),block_size, _block_crsMat.graph.row_map, _block_crsMat.graph.entries, _block_crsMat.values,
                  		z, r, true, true, 1.0, apply_count);

      //symmetric_gauss_seidel_apply(
      //    &kh,
      //    count_total, count_total,
      //    point_crsMat.graph.row_map,
      //    point_crsMat.graph.entries,
      //    point_crsMat.values, z, r, true,
      //    apply_count);

      Space().fence();
      precond_time += timer.seconds();
      precond_r_dot = KokkosBlas::dot(r , z );
      precond_beta  = precond_r_dot / precond_old_rdot ;
    }

    double beta  = 1;
    if (!use_sgs){
      beta = beta_original;
      // p = r + beta * p ;
      KokkosBlas::axpby(1.0, r, beta, p);
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

  Space().fence();
  iter_time = wall_clock.seconds();

  if ( 0 != result ) {
    result->iteration   = iteration ;
    result->iter_time   = iter_time ;
    result->matvec_time = matvec_time ;
    result->norm_res    = norm_res ;
    result->precond_time = precond_time;
    result->precond_init_time = precond_init_time;
  }

  if (use_sgs & owner_handle ){

    kh.destroy_gs_handle();
  }
}
*/


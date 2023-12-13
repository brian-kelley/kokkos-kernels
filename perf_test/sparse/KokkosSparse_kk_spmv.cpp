//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <cstdio>

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <cmath>
#include <unordered_map>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosKernels_IOUtils.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosSparse_spmv.hpp>
#include "KokkosKernels_default_types.hpp"

typedef double Scalar;
typedef int64_t Ordinal;
typedef size_t Offset;

void kokkos_sparse_kernel_0(
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::CudaSpace> v1 /* y */,
    Kokkos::View<const size_t*, Kokkos::LayoutRight, Kokkos::CudaSpace> v2 /* Arowptr */,
    Kokkos::View<const int64_t*, Kokkos::LayoutRight, Kokkos::CudaSpace> v3 /* Aentries */,
    Kokkos::View<const double*, Kokkos::LayoutRight, Kokkos::CudaSpace> v4 /* Avalues */,
    Kokkos::View<const double*, Kokkos::LayoutRight, Kokkos::CudaSpace> v5 /* x */) {
  using exec_space = Kokkos::DefaultExecutionSpace;
  size_t m = v2.extent(0) - 1;
  // scf.parallel
  typedef Kokkos::TeamPolicy<exec_space>::member_type member_type;
  int league_size = (m - 0 + 1 - 1) / 1;
  //Kokkos::TeamPolicy<exec_space> policy (league_size, Kokkos::AUTO(), Kokkos::AUTO() );
  Kokkos::TeamPolicy<exec_space> policy (league_size, 1, 8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(member_type member)
  {
    int64_t unit_v6 = member.league_rank ();
    int64_t v6 = 0 + unit_v6 * 1;
    // memref.load
    double v7 = v1(v6);
    // memref.load
    size_t v8 = v2(v6);
    // arith.addi
    size_t v9 = v6 + 1;
    // memref.load
    size_t v10 = v2(v9);
    // scf.parallel
    double v11;
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, (v10 - v8 + 1 - 1) / 1),
    [&](const int64_t &unit_v12, double& v13)
    {
      int64_t v12 = v8 + unit_v12 * 1;
      // memref.load
      size_t v14 = v3(v12);
      // memref.load
      double v15 = v4(v12);
      // memref.load
      double v16 = v5(v14);
      // arith.mulf
      double v17 = v15 * v16;
      // scf.reduce
      v13 += v17;
      // scf.yield
      ;
    }, v11)
    ;
    // memref.store
    Kokkos::single(Kokkos::PerTeam(member),
        [&] () { v1(v6) = v11; });
    // scf.yield
    ;
  })
  ;
  // func.return
  return;
}

void kokkos_sparse_kernel_1(Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::AnonymousSpace> v1, Kokkos::View<const size_t*, Kokkos::LayoutRight, Kokkos::AnonymousSpace> v2, Kokkos::View<const int64_t*, Kokkos::LayoutRight, Kokkos::AnonymousSpace> v3, Kokkos::View<const double*, Kokkos::LayoutRight, Kokkos::AnonymousSpace> v4, Kokkos::View<const double*, Kokkos::LayoutRight, Kokkos::AnonymousSpace> v5) {
  using exec_space = Kokkos::DefaultExecutionSpace;
  size_t m = v2.extent(0) - 1;
  // scf.parallel
  Kokkos::parallel_for(Kokkos::RangePolicy<exec_space>(0, (m - 0 + 1 - 1) / 1),
  KOKKOS_LAMBDA(int64_t unit_v6)
  {
    int64_t v6 = 0 + unit_v6 * 1;
    // memref.load
    double v7 = v1(v6);
    // memref.load
    size_t v8 = v2(v6);
    // arith.addi
    size_t v9 = v6 + 1;
    // memref.load
    size_t v10 = v2(v9);
    // scf.for
    double v11;
    double v12 = v7;
    for (size_t v13 = v8; v13 < v10; v13 += 1) {
      // memref.load
      size_t v14 = v3(v13);
      // memref.load
      double v15 = v4(v13);
      // memref.load
      double v16 = v5(v14);
      // arith.mulf
      double v17 = v15 * v16;
      // arith.addf
      double v18 = v12 + v17;
      v12 = v18;
    }
    v11 = v12;;
    // memref.store
    v1(v6) = v11;
    // scf.yield
    ;
  })
  ;
  // func.return
  return;
}

template <typename Layout>
void run_spmv(Ordinal numRows, Ordinal numCols, const char* filename, int loop,
              int num_vecs, char mode, Scalar beta) {
  typedef KokkosSparse::CrsMatrix<Scalar, Ordinal,
                                  Kokkos::DefaultExecutionSpace, void, Offset>
      matrix_type;
  typedef typename Kokkos::View<Scalar**, Layout> mv_type;
  typedef typename mv_type::HostMirror h_mv_type;

  srand(17312837);
  matrix_type A;
  if (filename)
    A = KokkosSparse::Impl::read_kokkos_crst_matrix<matrix_type>(filename);
  else {
    Offset nnz = 10 * numRows;
    // note: the help text says the bandwidth is fixed at 0.01 * numRows
    A = KokkosSparse::Impl::kk_generate_sparse_matrix<matrix_type>(
        numRows, numCols, nnz, 0, 0.01 * numRows);
  }
  numRows = A.numRows();
  numCols = A.numCols();

  std::cout << "A is " << numRows << "x" << numCols << ", with " << A.nnz()
            << " nonzeros\n";
  std::cout << "SpMV mode " << mode << ", " << num_vecs
            << " vectors, beta = " << beta << ", multivectors are ";
  std::cout << (std::is_same_v<Layout, Kokkos::LayoutLeft> ? "LayoutLeft"
                                                           : "LayoutRight");
  std::cout << '\n';

  mv_type x("X", numCols, num_vecs);
  mv_type y("Y", numRows, num_vecs);
  h_mv_type h_x         = Kokkos::create_mirror_view(x);
  h_mv_type h_y         = Kokkos::create_mirror_view(y);
  h_mv_type h_y_compare = Kokkos::create_mirror(y);

  for (int v = 0; v < num_vecs; v++) {
    for (int i = 0; i < numCols; i++) {
      h_x(i, v) = (Scalar)(1.0 * (rand() % 40) - 20.);
    }
  }

  Kokkos::deep_copy(x, h_x);

  // Benchmark
  auto x0 = Kokkos::subview(x, Kokkos::ALL(), 0);
  auto y0 = Kokkos::subview(y, Kokkos::ALL(), 0);
  // Do 5 warm up calls (not timed)
  for (int i = 0; i < 5; i++) {
    KokkosSparse::spmv(&mode, 1.0, A, x0, beta, y0);
    Kokkos::DefaultExecutionSpace().fence();
  }
  Kokkos::Timer timer;
  for (int i = 0; i < loop; i++) {
    KokkosSparse::spmv(&mode, 1.0, A, x0, beta, y0);
    Kokkos::DefaultExecutionSpace().fence();
  }
  double avg_time = timer.seconds() / loop;
  std::cout << "KK default (tpl): " << avg_time << " s\n";

  KokkosKernels::Experimental::Controls c({{"algorithm", "native"}});
  // Do 5 warm up calls (not timed)
  for (int i = 0; i < 5; i++) {
    KokkosSparse::spmv(c, &mode, 1.0, A, x0, beta, y0);
    Kokkos::DefaultExecutionSpace().fence();
  }
  timer.reset();
  for (int i = 0; i < loop; i++) {
    KokkosSparse::spmv(c, &mode, 1.0, A, x0, beta, y0);
    Kokkos::DefaultExecutionSpace().fence();
  }
  avg_time = timer.seconds() / loop;
  std::cout << "KK native: " << avg_time << " s\n";

  // Do 5 warm up calls (not timed)
//void kokkos_sparse_kernel_0(
//    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::AnonymousSpace> v1 /* y */,
//    Kokkos::View<size_t*, Kokkos::LayoutRight, Kokkos::AnonymousSpace> v2 /* Arowptr */,
//    Kokkos::View<size_t*, Kokkos::LayoutRight, Kokkos::AnonymousSpace> v3 /* Aentries */,
//    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::AnonymousSpace> v4 /* Avalues */,
//    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::AnonymousSpace> v5 /* x */) {
  for (int i = 0; i < 5; i++) {
    kokkos_sparse_kernel_1(y0, A.graph.row_map, A.graph.entries, A.values, x0);
    Kokkos::DefaultExecutionSpace().fence();
  }
  timer.reset();
  for (int i = 0; i < loop; i++) {
    kokkos_sparse_kernel_1(y0, A.graph.row_map, A.graph.entries, A.values, x0);
    Kokkos::DefaultExecutionSpace().fence();
  }
  avg_time = timer.seconds() / loop;
  std::cout << "MLIR: " << avg_time << " s\n";
}

void print_help() {
  printf("  -s [nrows]            : matrix dimension (square)\n");
  printf(
      "  --nv n                : number of columns in x/y multivector (default "
      "1).\n");
  printf(
      "  --layout left|right   : memory layout of x/y. Default depends on "
      "build's default execution space\n");
  printf(
      "  -m N|T                : matrix apply mode: N (normal, default), T "
      "(transpose)\n");
  printf(
      "  -f [file],-fb [file]  : Read in Matrix Market (.mtx), or binary "
      "(.bin) matrix file.\n");
  printf(
      "  -l [LOOP]             : How many spmv to run to aggregate average "
      "time. \n");
  printf("  -b beta               : beta, as in y := Ax + (beta)y\n");
}

int main(int argc, char** argv) {
  long long int size = 110503;  // a prime number
  char* filename     = NULL;

  char mode = 'N';
  char layout;
  if (std::is_same<default_layout, Kokkos::LayoutLeft>::value)
    layout = 'L';
  else
    layout = 'R';
  int loop     = 100;
  int num_vecs = 1;
  Scalar beta  = 0.0;

  if (argc == 1) {
    print_help();
    return 0;
  }

  for (int i = 0; i < argc; i++) {
    if ((strcmp(argv[i], "-s") == 0)) {
      size = atoi(argv[++i]);
      continue;
    }
    if ((strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "-fb") == 0)) {
      filename = argv[++i];
      continue;
    }
    if ((strcmp(argv[i], "-l") == 0)) {
      loop = atoi(argv[++i]);
      continue;
    }
    if ((strcmp(argv[i], "-m") == 0)) {
      mode = toupper(argv[++i][0]);
      continue;
    }
    if ((strcmp(argv[i], "--nv") == 0)) {
      num_vecs = atoi(argv[++i]);
      continue;
    }
    if ((strcmp(argv[i], "-b") == 0)) {
      beta = atof(argv[++i]);
      continue;
    }
    if ((strcmp(argv[i], "--layout") == 0)) {
      i++;
      if (toupper(argv[i][0]) == 'L')
        layout = 'L';
      else if (toupper(argv[i][0]) == 'R')
        layout = 'R';
      else
        throw std::runtime_error("Invalid layout");
    }
    if ((strcmp(argv[i], "--help") == 0) || (strcmp(argv[i], "-h") == 0)) {
      print_help();
      return 0;
    }
  }

  Kokkos::initialize(argc, argv);

  if (layout == 'L')
    run_spmv<Kokkos::LayoutLeft>(size, size, filename, loop, num_vecs, mode,
                                 beta);
  else
    run_spmv<Kokkos::LayoutRight>(size, size, filename, loop, num_vecs, mode,
                                  beta);

  Kokkos::finalize();
}

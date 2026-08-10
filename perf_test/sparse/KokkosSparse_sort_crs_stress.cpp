// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// Stress test for KokkosSparse::sort_crs_matrix targeting synchronization bugs
// (e.g., missing fences) on GPU backends such as AMD HIP.
//
// The test runs indefinitely, generating a new random CRS matrix each iteration
// with varying row count and a non-uniform per-row entry distribution, sorting
// it, and verifying that every row's column indices are non-decreasing.
// If a verification failure is detected the program prints the iteration number
// and exits with a non-zero status.  Stop it with Ctrl-C (SIGINT).
//
// Usage:
//   ./sparse_sort_crs_stress [--seed <N>] [--min-rows <N>] [--max-rows <N>]
//                            [--min-row-nnz <N>] [--max-row-nnz <N>]
//                            [--long-row-prob <0..1>] [--long-row-nnz <N>]
//
// All matrix views are allocated on Kokkos::DefaultExecutionSpace.

#include <Kokkos_Core.hpp>
#include <KokkosKernels_default_types.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_SortCrs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using scalar_t  = KokkosKernels::default_scalar;
using lno_t     = KokkosKernels::default_lno_t;
using size_type = KokkosKernels::default_size_type;

using exec_space   = Kokkos::DefaultExecutionSpace;
using device_t     = Kokkos::Device<exec_space, typename exec_space::memory_space>;
using crsMat_t     = KokkosSparse::CrsMatrix<scalar_t, lno_t, device_t, void, size_type>;

using scalar_view_t  = typename crsMat_t::values_type::non_const_type;
using entries_view_t = typename crsMat_t::index_type::non_const_type;
using rowmap_view_t  = typename crsMat_t::row_map_type::non_const_type;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
struct StressParams {
  uint64_t seed         = 0;       // 0 → use random device
  lno_t    minRows      = 1;
  lno_t    maxRows      = 5000;
  lno_t    minRowNnz    = 0;       // min entries per ordinary row
  lno_t    maxRowNnz    = 64;      // max entries per ordinary row
  double   longRowProb  = 0.02;    // probability that a row is a "long" row
  lno_t    longRowNnz   = 2048;    // entries in a long row
};

// ---------------------------------------------------------------------------
// Helper: read next CLI argument
// ---------------------------------------------------------------------------
static const char* nextArg(int& i, int argc, char** argv) {
  ++i;
  if (i >= argc) {
    std::cerr << "Error: expected value after " << argv[i - 1] << '\n';
    std::exit(1);
  }
  return argv[i];
}

// ---------------------------------------------------------------------------
// Generate a random, intentionally unsorted CRS matrix on host, then copy to
// device.  Row lengths follow a bimodal distribution controlled by longRowProb
// and longRowNnz to stress-test both short-row (thread-level) and long-row
// (bulk/bitonic) sort code paths simultaneously.
// ---------------------------------------------------------------------------
crsMat_t generateMatrix(const StressParams& p, std::mt19937_64& rng) {
  std::uniform_int_distribution<lno_t> rowDist(p.minRows, p.maxRows);
  const lno_t nrows = rowDist(rng);
  // ncols is chosen so that even the longest row can always be filled
  const lno_t ncols = std::max<lno_t>(p.longRowNnz * 2, p.maxRowNnz * 4);

  std::uniform_int_distribution<lno_t> nnzDist(p.minRowNnz, p.maxRowNnz);
  std::uniform_real_distribution<double> prob01(0.0, 1.0);

  // Build rowmap on host
  std::vector<size_type> rowmap(nrows + 1, 0);
  for (lno_t r = 0; r < nrows; ++r) {
    lno_t rowLen;
    if (prob01(rng) < p.longRowProb) {
      rowLen = p.longRowNnz;
    } else {
      rowLen = nnzDist(rng);
    }
    // Clamp to available columns (no duplicate columns allowed)
    if (rowLen > ncols) rowLen = ncols;
    rowmap[r + 1] = rowmap[r] + static_cast<size_type>(rowLen);
  }
  const size_type totalNnz = rowmap[nrows];

  // Build unsorted column indices on host.
  // For each row: sample rowLen distinct columns from [0, ncols) without
  // replacement, then do NOT sort them — that's the point.
  std::vector<lno_t> entries(totalNnz);
  // Reusable scratch for column index generation
  std::vector<lno_t> allCols(ncols);
  for (lno_t c = 0; c < ncols; ++c) allCols[c] = c;

  for (lno_t r = 0; r < nrows; ++r) {
    const size_type rowBegin = rowmap[r];
    const lno_t     rowLen   = static_cast<lno_t>(rowmap[r + 1] - rowBegin);
    if (rowLen == 0) continue;

    if (rowLen <= ncols / 2) {
      // For short rows: reservoir / rejection-sampling approach
      std::unordered_set<lno_t> chosen;
      chosen.reserve(rowLen * 2);
      std::uniform_int_distribution<lno_t> colDist(0, ncols - 1);
      for (lno_t k = 0; k < rowLen; ++k) {
        lno_t col;
        do { col = colDist(rng); } while (!chosen.insert(col).second);
        entries[rowBegin + k] = col;
      }
    } else {
      // For long rows: partial Fisher-Yates shuffle of allCols
      for (lno_t k = 0; k < rowLen; ++k) {
        std::uniform_int_distribution<lno_t> swapDist(k, ncols - 1);
        lno_t j = swapDist(rng);
        std::swap(allCols[k], allCols[j]);
        entries[rowBegin + k] = allCols[k];
      }
      // Restore swapped elements back (reset only touched range)
      // by re-initialising; cheaper than a full reset for large ncols
      for (lno_t k = 0; k < rowLen; ++k) allCols[k] = k;
    }
    // Leave the row unsorted — the sort function must order it.
  }

  // Values are irrelevant; fill with 1.0
  std::vector<scalar_t> values(totalNnz, static_cast<scalar_t>(1));

  // Copy to device views
  scalar_view_t  valView ("values",  totalNnz);
  entries_view_t entView ("entries", totalNnz);
  rowmap_view_t  rmView  ("rowmap",  nrows + 1);

  auto hVal = Kokkos::create_mirror_view(valView);
  auto hEnt = Kokkos::create_mirror_view(entView);
  auto hRm  = Kokkos::create_mirror_view(rmView);

  for (size_type i = 0; i < totalNnz; ++i) {
    hVal(i) = values[i];
    hEnt(i) = entries[i];
  }
  for (lno_t r = 0; r <= nrows; ++r) hRm(r) = rowmap[r];

  Kokkos::deep_copy(valView, hVal);
  Kokkos::deep_copy(entView, hEnt);
  Kokkos::deep_copy(rmView,  hRm);
  exec_space().fence();

  return crsMat_t("stress_matrix", nrows, ncols, totalNnz, valView, rmView, entView);
}

// ---------------------------------------------------------------------------
// Verify that every row is sorted (non-decreasing column indices).
// Returns true if sorted, false otherwise (and prints row/entry info).
// ---------------------------------------------------------------------------
bool verifySorted(const crsMat_t& A, long long iteration) {
  // Mirror back to host for verification
  auto hRm  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.row_map);
  auto hEnt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.entries);
  exec_space().fence();

  const lno_t nrows = A.numRows();
  for (lno_t r = 0; r < nrows; ++r) {
    const size_type rowBegin = hRm(r);
    const size_type rowEnd   = hRm(r + 1);
    for (size_type k = rowBegin + 1; k < rowEnd; ++k) {
      if (hEnt(k) < hEnt(k - 1)) {
        std::cerr << "[FAIL] Iteration " << iteration
                  << ": row " << r << " is not sorted at entry index " << k
                  << " (col[" << k-1 << "]=" << hEnt(k-1)
                  << " > col[" << k << "]=" << hEnt(k) << ")\n";
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  StressParams p;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--seed"))
      p.seed = static_cast<uint64_t>(std::atoll(nextArg(i, argc, argv)));
    else if (!std::strcmp(argv[i], "--min-rows"))
      p.minRows = static_cast<lno_t>(std::atoi(nextArg(i, argc, argv)));
    else if (!std::strcmp(argv[i], "--max-rows"))
      p.maxRows = static_cast<lno_t>(std::atoi(nextArg(i, argc, argv)));
    else if (!std::strcmp(argv[i], "--min-row-nnz"))
      p.minRowNnz = static_cast<lno_t>(std::atoi(nextArg(i, argc, argv)));
    else if (!std::strcmp(argv[i], "--max-row-nnz"))
      p.maxRowNnz = static_cast<lno_t>(std::atoi(nextArg(i, argc, argv)));
    else if (!std::strcmp(argv[i], "--long-row-prob"))
      p.longRowProb = std::atof(nextArg(i, argc, argv));
    else if (!std::strcmp(argv[i], "--long-row-nnz"))
      p.longRowNnz = static_cast<lno_t>(std::atoi(nextArg(i, argc, argv)));
    else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
      std::cout <<
        "Usage: sparse_sort_crs_stress [options]\n"
        "Options:\n"
        "  --seed <N>          RNG seed (0 = random device, default 0)\n"
        "  --min-rows <N>      minimum row count per matrix (default 1)\n"
        "  --max-rows <N>      maximum row count per matrix (default 5000)\n"
        "  --min-row-nnz <N>   minimum entries per short row (default 0)\n"
        "  --max-row-nnz <N>   maximum entries per short row (default 64)\n"
        "  --long-row-prob <F> probability a row becomes a long row (default 0.02)\n"
        "  --long-row-nnz <N>  entries in a long row (default 2048)\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << '\n';
      return 1;
    }
  }

  Kokkos::initialize(argc, argv);
  {
    // Seed the host RNG
    const uint64_t seed = (p.seed != 0) ? p.seed
                                        : static_cast<uint64_t>(
                                            std::random_device{}());
    std::mt19937_64 rng(seed);

    std::cout << "sort_crs_matrix stress test\n"
              << "  Execution space : " << exec_space().name() << '\n'
              << "  RNG seed        : " << seed << '\n'
              << "  Row count range : [" << p.minRows << ", " << p.maxRows << "]\n"
              << "  Short-row nnz   : [" << p.minRowNnz << ", " << p.maxRowNnz << "]\n"
              << "  Long-row prob   : " << p.longRowProb
              << " (nnz=" << p.longRowNnz << ")\n"
              << "Running until Ctrl-C or failure...\n\n";

    auto wallStart = std::chrono::steady_clock::now();
    long long iter = 0;

    while (true) {
      ++iter;

      // 1. Generate an unsorted CRS matrix on DefaultExecutionSpace
      crsMat_t A = generateMatrix(p, rng);

      // 2. Sort column indices in-place
      KokkosSparse::sort_crs_matrix(A);

      // 3. Fence to ensure sort kernel has completed before we read back
      exec_space().fence();

      // 4. Verify correctness
      if (!verifySorted(A, iter)) {
        std::cerr << "Stress test FAILED on iteration " << iter << '\n';
        Kokkos::finalize();
        return 1;
      }

      // 5. Progress report every 100 iterations
      if (iter % 100 == 0) {
        auto now     = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - wallStart).count();
        std::cout << "Iteration " << iter
                  << "  elapsed=" << elapsed << "s"
                  << "  iters/s=" << iter / elapsed
                  << "  last matrix: " << A.numRows() << " rows, "
                  << A.nnz() << " nnz\n";
        std::cout.flush();
      }
    }
  }
  Kokkos::finalize();
  return 0;
}

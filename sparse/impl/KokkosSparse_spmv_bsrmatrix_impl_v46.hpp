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

#ifndef KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_V46_HPP
#define KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_V46_HPP

#include <Kokkos_Core.hpp>

#include <KokkosKernels_ViewUtils.hpp>

namespace KokkosSparse {
namespace Impl {

/*
 * V46 GPU BSR spmv:
 * - Maximize parallelism: ideally one thread per scalar entry of A
 * - Maximize memory coalescing and global bandwidth utilization
 *     - reads from A and x should use all lanes to read consecutive values, and all values should be used.
 *     - updates to y should be coalesced too
 * - For each team, initiate A,x global reads as soon as possible to maximally saturate the memory system
 */
template <typename ExecSpace, typename Alpha, typename AMatrix, typename XVector, typename Beta, typename YVector>
struct BsrSpmvV46NonTrans {
  using size_type       = typename AMatrix::non_const_size_type;
  using Ordinal         = typename AMatrix::non_const_ordinal_type;
  using TeamPol         = Kokkos::TeamPolicy<ExecSpace>;
  using TeamMem         = typename TeamPol::member_type;
  using AScalar         = typename AMatrix::non_const_value_type;
  using XScalar         = typename XVector::non_const_value_type;
  using YScalar         = typename YVector::value_type;
  using ScratchSpace    = typename ExecSpace::scratch_memory_space;
  using IntScratch      = Kokkos::View<int *, ScratchSpace>;
  using SizeTypeScratch = Kokkos::View<size_type *, ScratchSpace>;
  using XScratch        = Kokkos::View<XScalar *, ScratchSpace>;
  using YScratch        = Kokkos::View<YScalar *, ScratchSpace>;

 public:
  BsrSpmvV46NonTrans(const AMatrix &A_, const XVector &x_, const YVector &y_, const Alpha &alpha_, const Beta &beta_)
      : A(A_), x(x_), y(y_), alpha(alpha_), beta(beta_) {
    // rowsPerTeam, entryChunkSize will be set after deciding the team size.
  }

  // How many bytes of per-team L0 scratch are required?
  int getScratchSize(int rowsPerTeam_, int entryChunkSize_, int bs_) {
    return XScratch::required_allocation_size(entryChunkSize_ * bs_) +
           YScratch::required_allocation_size(entryChunkSize_ * bs_) +
           YScratch::required_allocation_size(rowsPerTeam_ * bs_) +
           SizeTypeScratch::required_allocation_size(rowsPerTeam_ + 1) +
           IntScratch::required_allocation_size(entryChunkSize_);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const TeamMem &t) const {
    int bs = A.blockDim();
    // Declare scratch views upfront, in an order than minimizes padding for alignment.
    //
    // Store required x entries for the chunk in shared too, since they each get reused bs times
    XScratch xSlice(t.team_scratch(0), entryChunkSize * bs);
    // And store intermediate y results for each block in the chunk, so that we can efficiently reduce them later
    YScratch yChunks(t.team_scratch(0), entryChunkSize * bs);
    // Store the output entries of y in shared, until writing them out once at the end. Initially beta * y.
    YScratch ySlice(t.team_scratch(0), rowsPerTeam * bs);
    // Also load required entries of rowmap, since this will get reused
    SizeTypeScratch rowmapSlice(t.team_scratch(0), rowsPerTeam + 1);
    // entryToRow will determine which row, in range [0, rowsToProcess), each entry in the chunk corresponds to
    IntScratch entryToRow(t.team_scratch(0), entryChunkSize);
    Ordinal rowStart      = t.league_rank() * rowsPerTeam;
    Ordinal rowsToProcess = rowsPerTeam;
    if (rowStart + rowsToProcess > A.numRows()) rowsToProcess = A.numRows() - rowStart;
    // This pfor does 3 things in parallel by combining iteration spaces:
    // - populate ySlice
    // - populate rowmapSlice
    // - zero-initialize entryToRow
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, (rowsToProcess * bs) + (rowsToProcess + 1) + entryChunkSize),
                         [=](Ordinal i) {
                           if (i < rowsToProcess * bs) {
                             // Read entries into ySlice, or zero out if beta is zero
                             if (beta == Kokkos::ArithTraits<Beta>::zero()) {
                               ySlice(i) = Kokkos::ArithTraits<YScalar>::zero();
                             } else {
                               ySlice(i) = beta * y(rowStart * bs + i);
                             }
                           } else {
                             i -= rowsToProcess * bs;
                             if (i < rowsToProcess + 1) {
                               rowmapSlice(i) = A.graph.row_map(rowStart + i);
                             } else {
                               i -= (rowsToProcess + 1);
                               entryToRow(i) = 0;
                             }
                           }
                         });
    t.team_barrier();
    size_type totalEntries = rowmapSlice(rowsToProcess) - rowmapSlice(0);
    size_type rowmapSlice0 = rowmapSlice(0);
    // Process the entries in chunks until the full spmv for these rows is complete
    for (size_type entryChunk = 0; entryChunk < totalEntries; entryChunk += entryChunkSize) {
      int chunkLen = entryChunkSize;
      if (entryChunk + chunkLen > totalEntries) {
        chunkLen = totalEntries - entryChunk;
      }
      // Update entryToRow using rowmapSlice.
      // Use the fact that from chunk to chunk, the rows for each individual chunk can only increase monotonically.
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, chunkLen + chunkLen * bs), [=](int i) {
        if (i < chunkLen) {
          size_type entry = rowmapSlice0 + entryChunk + i;
          int row         = entryToRow(i);
          while (entry > rowmapSlice(row + 1)) row++;
          entryToRow(i) = row;
        } else {
          i -= chunkLen;
          int block       = i / bs;
          int colInBlock  = i % bs;
          size_type entry = rowmapSlice0 + entryChunk + block;
          Ordinal col     = A.graph.entries(entry);
          xSlice(i)       = x(col * bs + colInBlock);
          yChunks(i)      = Kokkos::ArithTraits<YScalar>::zero();
        }
      });
      t.team_barrier();
      // What is the starting index of this chunk within A's values?
      size_t chunkValueOffset = size_t(rowmapSlice0 + entryChunk) * bs * bs;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, chunkLen * bs * bs), [=](int i) {
        int work     = i;
        int blockCol = i % bs;
        i /= bs;
        int blockRow = i % bs;
        i /= bs;
        // Perfectly coalesced read from A
        AScalar aval = A.values(chunkValueOffset + work);
        // i is block index relative to entryChunk.
        // blockRow, blockCol give element within block.
        // work is the index of the scalar in A we will multiply.
        Kokkos::atomic_add(&yChunks(i * bs + blockRow), aval * xSlice(i * bs + blockCol));
      });
      t.team_barrier();
      // Then combine block-level results into ySlice
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, chunkLen * bs), [=](int i) {
        YScalar contribution = alpha * yChunks(i);
        int rowToUpdate      = entryToRow(i / bs);
        int blockRow         = i % bs;
        Kokkos::atomic_add(&ySlice(rowToUpdate * bs + blockRow), contribution);
      });
      t.team_barrier();
    }
    // Finally, write ySlice back out to y
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, rowsToProcess * bs),
                         [=](Ordinal i) { y(rowStart * bs + i) = ySlice(i); });
  }

  AMatrix A;
  XVector x;
  YVector y;
  Alpha alpha;
  Beta beta;
  int rowsPerTeam;
  int entryChunkSize;
};

template <typename ExecSpace, typename Alpha, typename AMatrix, typename XVector, typename Beta, typename YVector>
void apply_v46(const ExecSpace &exec, const Alpha &alpha, const AMatrix &A, const XVector &x, const Beta &beta,
               const YVector &y) {
  using TeamPol = Kokkos::TeamPolicy<ExecSpace>;

  static_assert(KokkosSparse::Experimental::is_bsr_matrix_v<AMatrix>,
                "SPMV_BSR_V46: AMatrix must be a KokkosSparse::BsrMatrix specialization.");

  // Make sure that the vectors are rank 1 (spmv already checked that xrank == yrank)
  static_assert(XVector::rank() == 1, "KokkosSparse::spmv: SPMV_BSR_V46 only supports rank-1 vectors currently");

  int bs = A.blockDim();
  // Use average blocks (nonzeros) per row to choose the team size
  size_t entriesPerRow = (A.nnz() + A.numRows() - 1) / A.numRows();
  size_t scalarsPerRow = entriesPerRow * bs * bs;
  BsrSpmvV46NonTrans<ExecSpace, Alpha, AMatrix, XVector, Beta, YVector> functor(A, x, y, alpha, beta);
  // Want team size to be as large as possible.
  // First decide what the maximum team size is, without factoring in scratch requirement
  TeamPol tempPolicy(1, 1);
  int teamSize       = tempPolicy.team_size_recommended(functor, Kokkos::ParallelForTag{});
  int scratchMax     = tempPolicy.scratch_size_max(0);
  size_t rowsPerTeam = teamSize / scalarsPerRow;
  if (rowsPerTeam == 0) rowsPerTeam = 1;
  size_t entryChunkSize = entriesPerRow * rowsPerTeam;
  // Determine the shared requirement in the optimal case
  int scratchRequired = functor.getScratchSize(rowsPerTeam, entryChunkSize, bs);
  while (scratchRequired > scratchMax - 128) {
    // Scratch requirement is too large; decrease the entryChunkSize until it fits
    entryChunkSize *= 0.8;
    scratchRequired = functor.getScratchSize(rowsPerTeam, entryChunkSize, bs);
  }
  TeamPol policy((A.numRows() + rowsPerTeam - 1) / rowsPerTeam, teamSize);
  policy.set_scratch_size(0, Kokkos::PerTeam(scratchRequired));
  functor.rowsPerTeam    = rowsPerTeam;
  functor.entryChunkSize = entryChunkSize;
  std::cout << "Launching bsr spmv v46.\n";
  std::cout << "rowsPerTeam = " << rowsPerTeam << '\n';
  std::cout << "entryChunkSize = " << entryChunkSize << '\n';
  std::cout << "scratchRequired = " << scratchRequired << '\n';
  std::cout << "teamSize = " << teamSize << '\n';
  std::cout << "scalars per team = " << scalarsPerRow * rowsPerTeam << '\n';
  Kokkos::parallel_for("KokkosSparse::spmv[bsr,v46]", policy, functor);
}

}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_V46_HPP

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

template <typename ExecSpace, typename Alpha, typename AMatrix, typename XVector, typename Beta, typename YVector, typename TaskRows>
struct BsrSpmvV46NonTrans {
  using size_type       = typename AMatrix::non_const_size_type;
  using Ordinal         = typename AMatrix::non_const_ordinal_type;
  using OrdinalMaxScan  = MaxScan<Ordinal>;
  using TeamPol         = Kokkos::TeamPolicy<ExecSpace>;
  using TeamMem         = typename TeamPol::member_type;
  using AScalar         = typename AMatrix::non_const_value_type;
  using XScalar         = typename XVector::non_const_value_type;
  using YScalar         = typename YVector::value_type;
  using ScratchSpace    = typename ExecSpace::scratch_memory_space;
  using OrdinalScratch  = Kokkos::View<Ordinal*, ScratchSpace>;
  using IntScratch      = Kokkos::View<int *, ScratchSpace>;
  using SizeTypeScratch = Kokkos::View<size_type *, ScratchSpace>;
  using XScratch        = Kokkos::View<XScalar *, ScratchSpace>;
  using YScratch        = Kokkos::View<YScalar *, ScratchSpace>;

  BsrSpmvV46NonTrans(const AMatrix &A_, const XVector &x_, const YVector &y_, const Alpha &alpha_, const Beta &beta_, const TaskRows& taskRows_)
      : A(A_), x(x_), y(y_), alpha(alpha_), beta(beta_), taskRows(taskRows_) {}

  KOKKOS_INLINE_FUNCTION void operator()(const TeamMem &t) const {
    // Allocate scratch views upfront.
    // products: store all intermediate scalar products
    YScratch products(t.team_scratch(0), t.team_size());
    // xvals: store all entries of x that this team will need
    XScratch xvals(t.team_scratch(0), nentries * bs);
    // rows: will map (block) entries to (block) rows.
    // For example, rows(0) will give the row containing the first block entry this team processes.
    // This avoid storing per-row information; a long run of empty rows could make that problematic.
    OrdinalScratch rows(t.team_scratch(0), nentries);
    // offsetInRow: index of an entry within its row
    // Only the entries processed by the team are counted, so the first element is 0
    // even if the team starts in the middle of a row.
    IntScratch offsetInRow(t.team_scratch(0), nentries);

    // At which scalar entry will this team starting reading from A?
    size_t teamScalarEntryBegin = size_t(t.league_rank()) * t.team_size();
    // One past last entry to read from A
    size_t teamScalarEntryEnd = teamScalarEntryBegin + t.team_size();
    if(teamScalarEntryEnd > A.values.extent(0))
      teamScalarEntryEnd = A.values.extent(0);
    // taskRows was pre-populated with the rows where each team's entry range starts.
    Ordinal rowStart = taskRows(t.league_rank());
    Ordinal rowEnd = taskRows(t.league_rank() + 1);
    // Read the required x values into shared (coalesced over block and across blocks if 
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, (teamEntryEnd - teamEntryBegin) * bs),
        [=](size_type i) {
          size_type entry = teamEntryBegin + (i / bs);
          size_type scalarInBlock = i % bs;
          Ordinal col = scalarInBlock + bs * A.graph.entries(entry);
          xvals(i) = x(col);
        });
    // Which block entry is the first one read by this team
    size_type teamEntryBegin = teamScalarEntryBegin / bs / bs;
    // One past the last block entry this team reads
    size_type teamEntryEnd = (teamScalarEntryEnd + bs * bs - 1) / bs / bs;
    // Compute entry -> row mapping.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, teamEntryEnd - teamEntryBegin),
        [=](int i) {
          rows(i) = 0;
        });
    t.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, rowEnd - rowBegin),
        [=](Ordinal i) {
          // Count the new rows which start at entry i (this is correct in the case of empty rows)
          // Do not include a row which beings exactly at teamEntryBegin, if there is one.
          // This would cause all the row indices to be 1 higher than they should be.
          size_type rowIBegin = A.graph.row_map(rowBegin + i);
          if(rowIBegin != teamEntryBegin)
            Kokkos::atomic_increment(&rows(rowIBegin - teamEntryBegin));
        });
    t.team_barrier();
    // Finally, do an exclusive scan over rows to figure out the actual rows of each entry (relative to rowBegin)
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, teamEntryEnd - teamEntryBegin),
        [=](Ordinal i, Ordinal& lrow, bool finalPass)
        {
          Ordinal val = rows(i);
          if(finalPass)
            rows(i) = lrow;
          lrow += val;
        });
    // Using entry -> row mapping, a segmented scan can populate offsetInRow.
    // The value to prefix-sum is a constant 1, and the segment is given by the row.
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, teamEntryEnd - teamEntryBegin),
        [=](Ordinal i, Ordinal& lrow, bool finalPass)
        {
          Ordinal val = rows(i);
          if(finalPass)
            rows(i) = lrow;
          lrow += val;
        });
    // Compute the intermediate products.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, teamScalarEntryEnd - teamScalarEntryBegin),
        [=](int i)
        {
          size_t Aindex = teamScalarEntryBegin + i;
          // Aligned + coalesced read from A.values
          AScalar Aval = A.values(Aindex);
          int blockCol = Aindex % bs;
          int blockRow = (Aindex / bs) % bs;
          int entry = Aindex / bs / bs - teamEntryBegin;
          // Write the intermediate products into shared so that values in each row are consecutive.
          products(entry * bs * bs ) = Aval * xvals(entry * bs = blockCol);
        });
    t.team_barrier();
    // Now that we have all the intermediate products, do a segmented reduction to sum across rows.
  }

  AMatrix A;
  XVector x;
  YVector y;
  Alpha alpha;
  Beta beta;
  TaskRows taskRows;
  // How many entries (upper bound) will this team process?
  // Note: strictly less than team size, so will always fit in int.
  int bs;
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
  std::cout << "alpha = " << alpha << '\n';
  std::cout << "beta = " << beta << '\n';
  std::cout << "rowsPerTeam = " << rowsPerTeam << '\n';
  std::cout << "entryChunkSize = " << entryChunkSize << '\n';
  std::cout << "scratchRequired = " << scratchRequired << '\n';
  std::cout << "teamSize = " << teamSize << '\n';
  std::cout << "scalars per team = " << scalarsPerRow * rowsPerTeam << '\n';
  std::cout << "\n\n";
  Kokkos::parallel_for("KokkosSparse::spmv[bsr,v46]", policy, functor);
}

}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_V46_HPP

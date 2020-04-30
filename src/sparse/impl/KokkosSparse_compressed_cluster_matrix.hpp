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
// Questions? Contact Brian Kelley (bmkelle@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include "KokkosKernels_Utils.hpp"

namespace KokkosSparse{
namespace Impl{

//if compactScalar is true, store matrix with 32-bit precision even if scalar is 64-bit
template<bool compact, typename scalar_t>
struct get_compact_scalar
{
  using type = scalar_t;
};

template<>
struct get_compact_scalar<true, double>
{
  using type = float;
};

template<>
struct get_compact_scalar<true, Kokkos::complex<double>>
{
  using type = Kokkos::complex<float>;
};

template <bool compactScalar, typename CGSHandle>
struct ClusterCompression
{
  using mem_space = typename CGSHandle::mem_space;
  using size_type = typename CGSHandle::size_type;
  using lno_t = typename CGSHandle::lno_t;
  using scalar_t = typename CGSHandle::scalar_t;
  using comp_scalar_t = typename get_compact_scalar<compactScalar, scalar_t>::type;
  using unit_t = typename CGSHandle::unit_t;
  using unit_view_t = typename CGSHandle::unit_view_t;
  using offset_view_t = typename CGSHandle::offset_view_t; 
  using ordinal_view_t = typename CGSHandle::ordinal_view_t; 
  //the input matrix to compress is accessed through const-valued views
  using rowmap_t = typename CGSHandle::const_rowmap_t;
  using entries_t = typename CGSHandle::const_entries_t;
  using values_t = typename CGSHandle::const_values_t;
  //will pad each overall cluster to this alignment
  static constexpr size_t memStreamAlign = alignof(lno_t) > alignof(comp_scalar_t) ? alignof(lno_t) : alignof(comp_scalar_t);
  //The sizes of both lno_t and comp_scalar_t should be multiples of unit_t's size
  static_assert(sizeof(lno_t) % sizeof(unit_t) == 0,
      "Expect lno_t's size to be a multiple of unit's size. Make ClusterGaussSeidelHandle::unit_t a smaller integer type.");
  static_assert(sizeof(comp_scalar_t) % sizeof(unit_t) == 0,
      "Expect compressed scalar type's size to be a multiple of unit's size. Make ClusterGaussSeidelHandle::unit_t a smaller integer type.");

  //Functor to compute the compressed size of each cluster
  //Also reorders clusters to be grouped by color
  struct CompressedSizeFunctor
  {
    CompressedSizeFunctor(
        const rowmap_t& rowmap_, const entries_t& entries_,
        const entries_t& clusterOffsets_, const entries_t& clusterVerts_,
        const entries_t& colorSets_,
        const offset_view_t& storageSize_)
      : rowmap(rowmap_), entries(entries_), clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), colorSets(colorSets_), storageSize(storageSize_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t dstC) const
    {
      //srcC is the real cluster (corresponding to clusterOffsets and clusterVerts)
      lno_t srcC = colorSets(dstC);
      lno_t clusterBegin = clusterOffsets(srcC);
      lno_t clusterEnd = clusterOffsets(srcC + 1);
      lno_t clusterSize = clusterEnd - clusterBegin;
      KokkosKernels::Impl::MemStream<unit_t> block;
      //1. store #rows in cluster
      block.template allocSingle<lno_t>();
      //2. store the list of rows in the cluster
      block.template getArray<lno_t>(clusterSize);
      //3. store the number of entries in each row (excluding diagonals)
      block.template getArray<lno_t>(clusterSize);
      //4. for each row: inverse diagonal, entries and values per row.
      //                 the first row must be scalar-aligned,
      //                 and each values array is also aligned.
      for(lno_t i = clusterBegin; i < clusterEnd; i++)
      {
        lno_t row = clusterVerts(i);
        //space for the inverse diagonal
        block.template allocSingle<comp_scalar_t>();
        //count off-diagonal entries to store
        int numOffDiag = 0;
        for(size_type j = rowmap(row); j < rowmap(row + 1); j++)
        {
          if(entries(j) != row)
            numOffDiag++;
        }
        //space for entries
        block.template getArray<lno_t>(numOffDiag);
        //space for off-diag columns
        block.template getArray<comp_scalar_t>(numOffDiag);
      }
      //finally, make sure that the storage is aligned to both comp_scalar_t and lno_t
      storageSize(dstC) = block.sizeInUnits(memStreamAlign);
      //one thread sets the size of the one-past-end cluster to 0, so that the counts
      //array doesn't need initialization. Do exclusive prefix sum like with CRS.
      if(dstC == lno_t(storageSize.extent(0) - 2))
        storageSize(storageSize.extent(0) - 1) = 0;
    }

    rowmap_t rowmap;
    entries_t entries;
    entries_t clusterOffsets;
    entries_t clusterVerts;
    entries_t colorSets;        //aka color_adj
    offset_view_t storageSize;    //output: the number of unit_t needed to store each cluster
  };

  struct PermutedCompressedSizeFunctor
  {
    PermutedCompressedSizeFunctor(
        const rowmap_t& rowmap_, const entries_t& entries_,
        const entries_t& clusterOffsets_, const entries_t& clusterVerts_,
        const entries_t& colorSets_,
        const offset_view_t& storageSize_)
      : rowmap(rowmap_), entries(entries_), clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), colorSets(colorSets_), storageSize(storageSize_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t dstC) const
    {
      //srcC is the real cluster (corresponding to clusterOffsets and clusterVerts)
      lno_t srcC = colorSets(dstC);
      lno_t clusterBegin = clusterOffsets(srcC);
      lno_t clusterEnd = clusterOffsets(srcC + 1);
      lno_t clusterSize = clusterEnd - clusterBegin;
      KokkosKernels::Impl::MemStream<unit_t> block;
      //1. store #rows in cluster
      block.template allocSingle<lno_t>();
      //2. store the offset of first row in cluster
      block.template allocSingle<lno_t>();
      //3. store the number of entries in each row (excluding diagonals)
      block.template getArray<lno_t>(clusterSize);
      //4. for each row: inverse diagonal, entries and values per row.
      //                 the first row must be scalar-aligned,
      //                 and each values array is also aligned.
      for(lno_t i = clusterBegin; i < clusterEnd; i++)
      {
        lno_t row = clusterVerts(i);
        //space for the inverse diagonal
        block.template allocSingle<comp_scalar_t>();
        //count off-diagonal entries to store
        int numOffDiag = 0;
        for(size_type j = rowmap(row); j < rowmap(row + 1); j++)
        {
          if(entries(j) != row)
            numOffDiag++;
        }
        //space for entries
        block.template getArray<lno_t>(numOffDiag);
        //space for off-diag columns
        block.template getArray<comp_scalar_t>(numOffDiag);
      }
      //finally, make sure that the storage is aligned to both comp_scalar_t and lno_t
      storageSize(dstC) = block.sizeInUnits(memStreamAlign);
      //one thread sets the size of the one-past-end cluster to 0, so that the counts
      //array doesn't need initialization. Do exclusive prefix sum like with CRS.
      if(dstC == lno_t(storageSize.extent(0) - 2))
        storageSize(storageSize.extent(0) - 1) = 0;
    }

    rowmap_t rowmap;
    entries_t entries;
    entries_t clusterOffsets;
    entries_t clusterVerts;
    entries_t colorSets;        //aka color_adj
    offset_view_t storageSize;    //output: the number of unit_t needed to store each cluster
  };

  //Functor to compute the compressed representation of clusters
  struct CompressFunctor
  {
    CompressFunctor(
        const rowmap_t& rowmap_, const entries_t& entries_, const values_t& values_,
        const entries_t& clusterOffsets_, const entries_t& clusterVerts_,
        const entries_t& colorSets_,
        const offset_view_t& compressedOffsets_, const unit_view_t& compressed_)
      : rowmap(rowmap_), entries(entries_), values(values_),
      clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_),
      colorSets(colorSets_), compressedOffsets(compressedOffsets_), compressed(compressed_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t dstC) const
    {
      //srcC is the real cluster (corresponding to clusterOffsets and clusterVerts)
      lno_t srcC = colorSets(dstC);
      lno_t clusterBegin = clusterOffsets(srcC);
      lno_t clusterEnd = clusterOffsets(srcC + 1);
      lno_t clusterSize = clusterEnd - clusterBegin;
      //lno_t and comp_scalar_t both always have power-of-two sizes,
      //so one size is always a multiple of the other.
      //Simulate the memory layout with a pointer to unit_t.
      KokkosKernels::Impl::MemStream<unit_t>
        block(&compressed(compressedOffsets(dstC)));
      //1. store #rows in cluster
      block.template writeSingle<lno_t>(clusterSize);
      //2. store the list of rows in the cluster
      lno_t* clusterRows = block.template getArray<lno_t>(clusterSize);
      for(lno_t i = 0; i < clusterSize; i++)
      {
        clusterRows[i] = clusterVerts(clusterBegin + i);
      }
      //3. store the number of entries in each row (excluding diagonals)
      lno_t* rowSizes = block.template getArray<lno_t>(clusterSize);
      for(lno_t i = 0; i < clusterSize; i++)
      {
        lno_t row = clusterRows[i];
        int numOffDiag = 0;
        for(size_type j = rowmap(row); j < rowmap(row + 1); j++)
        {
          if(entries(j) != row)
            numOffDiag++;
        }
        rowSizes[i] = numOffDiag;
      }
      //4. for each row: inverse diagonal, entries and values per row.
      //                 the first row must be scalar-aligned,
      //                 and each values array is also aligned.
      for(lno_t i = 0; i < clusterSize; i++)
      {
        lno_t row = clusterRows[i];
        //divide up storage for the row:
        //determine where diag^-1, entries, values will go
        comp_scalar_t* invDiag = block.template getArray<comp_scalar_t>(1);
        lno_t* rowEntries = block.template getArray<lno_t>(rowSizes[i]);
        comp_scalar_t* rowValues = block.template getArray<comp_scalar_t>(rowSizes[i]);
        int numOffDiag = 0;
        for(size_type j = rowmap(row); j < rowmap(row + 1); j++)
        {
          if(entries(j) == row)
          {
            *invDiag =
              Kokkos::ArithTraits<comp_scalar_t>::one() / comp_scalar_t(values(j));
          }
          else
          {
            rowEntries[numOffDiag] = entries(j);
            rowValues[numOffDiag] = comp_scalar_t(values(j));
            numOffDiag++;
          }
        }
      }
    }

    rowmap_t rowmap;
    entries_t entries;
    values_t values;
    entries_t clusterOffsets;
    entries_t clusterVerts;
    entries_t colorSets;
    //offset of each cluster in compressed
    offset_view_t compressedOffsets;
    //output: the compact representation of all clusters
    unit_view_t compressed;
  };

  //Functor to compute the compressed representation of clusters
  struct PermutedCompressFunctor
  {
    PermutedCompressFunctor(
        const rowmap_t& rowmap_, const entries_t& entries_, const values_t& values_,
        const entries_t& clusterOffsets_, const entries_t& clusterVerts_,
        const entries_t& colorSets_,
        const ordinal_view_t& invPerm_,
        const offset_view_t& compressedOffsets_, const unit_view_t& compressed_)
      : rowmap(rowmap_), entries(entries_), values(values_),
      clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_),
      colorSets(colorSets_),
      invPerm(invPerm_), numRows(invPerm.extent(0)),
      compressedOffsets(compressedOffsets_), compressed(compressed_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t dstC) const
    {
      //srcC is the real cluster (corresponding to clusterOffsets and clusterVerts)
      lno_t srcC = colorSets(dstC);
      lno_t clusterBegin = clusterOffsets(srcC);
      lno_t clusterEnd = clusterOffsets(srcC + 1);
      lno_t clusterSize = clusterEnd - clusterBegin;
      //lno_t and comp_scalar_t both always have power-of-two sizes,
      //so one size is always a multiple of the other.
      //Simulate the memory layout with a pointer to unit_t.
      KokkosKernels::Impl::MemStream<unit_t>
        block(&compressed(compressedOffsets(dstC)));
      //1. store #rows in cluster
      block.template writeSingle<lno_t>(clusterSize);
      //2. store the first row (permuted) in the cluster
      lno_t firstRowInput = clusterVerts(clusterBegin);
      lno_t firstRowPerm = invPerm(firstRowInput);
      block.template writeSingle<lno_t>(firstRowPerm);
      //3. store the number of entries in each row (excluding diagonals)
      lno_t* rowSizes = block.template getArray<lno_t>(clusterSize);
      for(lno_t i = 0; i < clusterSize; i++)
      {
        lno_t row = clusterVerts(clusterBegin + i);
        int numOffDiag = 0;
        for(size_type j = rowmap(row); j < rowmap(row + 1); j++)
        {
          if(entries(j) != row)
            numOffDiag++;
        }
        rowSizes[i] = numOffDiag;
      }
      //4. for each row: inverse diagonal, entries and values per row.
      //                 the first row must be scalar-aligned,
      //                 and each values array is also aligned.
      for(lno_t i = 0; i < clusterSize; i++)
      {
        lno_t row = clusterVerts(clusterBegin + i);
        //divide up storage for the row:
        //determine where diag^-1, entries, values will go
        comp_scalar_t* invDiag = block.template getArray<comp_scalar_t>(1);
        lno_t* rowEntries = block.template getArray<lno_t>(rowSizes[i]);
        comp_scalar_t* rowValues = block.template getArray<comp_scalar_t>(rowSizes[i]);
        int numOffDiag = 0;
        for(size_type j = rowmap(row); j < rowmap(row + 1); j++)
        {
          if(entries(j) == row)
          {
            *invDiag =
              Kokkos::ArithTraits<comp_scalar_t>::one() / comp_scalar_t(values(j));
          }
          else
          {
            lno_t col = entries(j);
            if(col < numRows)
              col = invPerm(col);
            //col is now the correct index for permuted x
            rowEntries[numOffDiag] = col;
            rowValues[numOffDiag] = comp_scalar_t(values(j));
            numOffDiag++;
          }
        }
      }
    }

    rowmap_t rowmap;
    entries_t entries;
    values_t values;
    entries_t clusterOffsets;
    entries_t clusterVerts;
    entries_t colorSets;
    ordinal_view_t invPerm;
    lno_t numRows;
    //offset of each cluster in compressed
    offset_view_t compressedOffsets;
    //output: the compact representation of all clusters
    unit_view_t compressed;
  };
};

template <bool compactScalar, typename CGSHandle, typename X_t_, typename Y_t_>
struct CompressedClusterApply
{
  using exec_space = typename CGSHandle::exec_space;
  using mem_space = typename CGSHandle::mem_space;
  using size_type = typename CGSHandle::size_type;
  using lno_t = typename CGSHandle::lno_t;
  using scalar_t = typename CGSHandle::scalar_t;
  using comp_scalar_t = typename get_compact_scalar<compactScalar, scalar_t>::type;
  using unit_t = typename CGSHandle::unit_t;
  using unit_view_t = typename CGSHandle::unit_view_t;
  using offset_view_t = typename CGSHandle::offset_view_t; 
  //the input matrix to compress is accessed through const-valued views
  using rowmap_t = typename CGSHandle::const_rowmap_t;
  using entries_t = typename CGSHandle::const_entries_t;
  using values_t = typename CGSHandle::const_values_t;
  using team_policy_t = Kokkos::TeamPolicy<exec_space>;
  using team_member_t = typename team_policy_t::member_type;
  //These are just so that X_t and Y_T are available as member typedefs
  using X_t = X_t_;
  using Y_t = Y_t_;

  //Range-policy version of apply (one row = one work item)
  struct ApplyRange
  {
    ApplyRange(
        const offset_view_t& compressedOffsets_, const unit_view_t& compressed_,
        const X_t& x_, const Y_t& y_, scalar_t omega_)
      : compressedOffsets(compressedOffsets_), compressed(compressed_), x(x_), y(y_), omega(omega_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t c) const
    {
      lno_t num_vecs = x.extent(1);
      //compressed layout is contiguous, and range policy will just be over the
      //color set, so no need to map the work-item index to a cluster.
      KokkosKernels::Impl::MemStream<unit_t>
        block(&compressed(compressedOffsets(c)));
      //1. get #rows in cluster
      lno_t clusterSize = block.template readSingle<lno_t>();
      //2. get the list of rows in the cluster
      lno_t* clusterRows = block.template getArray<lno_t>(clusterSize);
      //3. store the number of entries in each row (excluding diagonals)
      lno_t* rowSizes = block.template getArray<lno_t>(clusterSize);
      //4. for each row: inverse diagonal, entries and values per row.
      //                 the first row must be scalar-aligned,
      //                 and each values array is also aligned.
      for(lno_t i = 0; i < clusterSize; i++)
      {
        lno_t row = clusterRows[i];
        lno_t rowSize = rowSizes[i];
        //divide up storage for the row:
        //determine where diag^-1, entries, values will go
        comp_scalar_t invDiag = block.template readSingle<comp_scalar_t>();
        lno_t* rowEntries = block.template getArray<lno_t>(rowSize);
        comp_scalar_t* rowValues = block.template getArray<comp_scalar_t>(rowSize);
        constexpr lno_t colBatchSize = 8;
        scalar_t accum[colBatchSize];
        scalar_t k = omega * invDiag;
        for(lno_t batch_start = 0; batch_start < num_vecs; batch_start += colBatchSize)
        {
          lno_t batch = colBatchSize;
          if(batch_start + batch > num_vecs)
            batch = num_vecs - batch_start;
          //the current batch of columns given by: batch_start, this_batch_size
          for(lno_t i = 0; i < batch; i++)
            accum[i] = y(row, batch_start + i);
          for(lno_t i = 0; i < rowSize; i++)
          {
            lno_t col = rowEntries[i];
            scalar_t val = rowValues[i];
            for(lno_t i = 0; i < batch; i++)
              accum[i] -= val * x(col, batch_start + i);
          }
          for(lno_t i = 0; i < batch; i++)
          {
            scalar_t newXval = x(row, batch_start + i) * (Kokkos::ArithTraits<scalar_t>::one() - omega);
            x(row, batch_start + i) = newXval + k * accum[i];
          }
        }
      }
    }

    //offset of each cluster in compressed
    offset_view_t compressedOffsets;
    //output: the compact representation of all clusters
    unit_view_t compressed;
    X_t x;
    Y_t y;
    scalar_t omega;
  };

  //Permuted range-policy version of apply
  struct PermuteApplyRange
  {
    PermuteApplyRange(
        const offset_view_t& compressedOffsets_, const unit_view_t& compressed_,
        const X_t& x_, const Y_t& y_, scalar_t omega_)
      : compressedOffsets(compressedOffsets_), compressed(compressed_), x(x_), y(y_), omega(omega_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t c) const
    {
      lno_t num_vecs = x.extent(1);
      //compressed layout is contiguous, and range policy will just be over the
      //color set, so no need to map the work-item index to a cluster.
      KokkosKernels::Impl::MemStream<unit_t>
        block(&compressed(compressedOffsets(c)));
      //1. get #rows in cluster
      lno_t clusterSize = block.template readSingle<lno_t>();
      //2. get the list of rows in the cluster
      lno_t clusterRowBegin = block.template readSingle<lno_t>();
      //3. store the number of entries in each row (excluding diagonals)
      lno_t* rowSizes = block.template getArray<lno_t>(clusterSize);
      //4. for each row: inverse diagonal, entries and values per row.
      //                 the first row must be scalar-aligned,
      //                 and each values array is also aligned.
      for(lno_t i = 0; i < clusterSize; i++)
      {
        lno_t row = clusterRowBegin + i;
        lno_t rowSize = rowSizes[i];
        //divide up storage for the row:
        //determine where diag^-1, entries, values will go
        comp_scalar_t invDiag = block.template readSingle<comp_scalar_t>();
        lno_t* rowEntries = block.template getArray<lno_t>(rowSize);
        comp_scalar_t* rowValues = block.template getArray<comp_scalar_t>(rowSize);
        constexpr lno_t colBatchSize = 8;
        scalar_t accum[colBatchSize];
        scalar_t k = omega * invDiag;
        for(lno_t batch_start = 0; batch_start < num_vecs; batch_start += colBatchSize)
        {
          lno_t batch = colBatchSize;
          if(batch_start + batch > num_vecs)
            batch = num_vecs - batch_start;
          //the current batch of columns given by: batch_start, this_batch_size
          for(lno_t i = 0; i < batch; i++)
            accum[i] = y(row, batch_start + i);
          for(lno_t i = 0; i < rowSize; i++)
          {
            lno_t col = rowEntries[i];
            scalar_t val = rowValues[i];
            for(lno_t i = 0; i < batch; i++)
              accum[i] -= val * x(col, batch_start + i);
          }
          for(lno_t i = 0; i < batch; i++)
          {
            scalar_t newXval = x(row, batch_start + i) * (Kokkos::ArithTraits<scalar_t>::one() - omega);
            x(row, batch_start + i) = newXval + k * accum[i];
          }
        }
      }
    }

    //offset of each cluster in compressed
    offset_view_t compressedOffsets;
    //output: the compact representation of all clusters
    unit_view_t compressed;
    X_t x;
    Y_t y;
    scalar_t omega;
  };

  //Team-policy version of apply (one row = one work item)
  //Set colorSetBegin and colorSetEnd before executing over each color set
  struct ApplyTeam
  {
    ApplyTeam(
        const offset_view_t& compressedOffsets_, const unit_view_t& compressed_,
        const X_t& x_, const Y_t& y_, scalar_t omega_)
      : compressedOffsets(compressedOffsets_), compressed(compressed_),
      x(x_), y(y_), omega(omega_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const team_member_t t) const
    {
      lno_t num_vecs = x.extent(1);
      lno_t teamClusterBegin = colorSetBegin + t.league_rank() * t.team_size();
      lno_t teamClusterEnd = teamClusterBegin + t.team_size();
      if(teamClusterEnd > colorSetEnd)
        teamClusterEnd = colorSetEnd;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, teamClusterEnd - teamClusterBegin),
        [&](const lno_t threadWork)
        {
          lno_t c = teamClusterBegin + threadWork;
          //compressed layout is contiguous, and range policy will just be over the
          //color set, so no need to map the work-item index to a cluster.
          KokkosKernels::Impl::MemStream<unit_t>
            block(&compressed(compressedOffsets(c)));
          //1. get #rows in cluster
          lno_t clusterSize = block.template readSingle<lno_t>();
          //2. get the list of rows in the cluster
          lno_t* clusterRows = block.template getArray<lno_t>(clusterSize);
          //3. store the number of entries in each row (excluding diagonals)
          lno_t* rowSizes = block.template getArray<lno_t>(clusterSize);
          //4. for each row: inverse diagonal, entries and values per row.
          //                 the first row must be scalar-aligned,
          //                 and each values array is also aligned.
          for(lno_t i = 0; i < clusterSize; i++)
          {
            lno_t row = clusterRows[i];
            lno_t rowSize = rowSizes[i];
            //divide up storage for the row:
            //determine where diag^-1, entries, values will go
            comp_scalar_t invDiag = block.template readSingle<comp_scalar_t>();
            lno_t* rowEntries = block.template getArray<lno_t>(rowSize);
            comp_scalar_t* rowValues = block.template getArray<comp_scalar_t>(rowSize);
            constexpr lno_t colBatchSize = 8;
            scalar_t accum[colBatchSize];
            scalar_t k = omega * invDiag;
            for(lno_t batch_start = 0; batch_start < num_vecs; batch_start += colBatchSize)
            {
              lno_t batch = colBatchSize;
              if(batch_start + batch > num_vecs)
                batch = num_vecs - batch_start;
              //the current batch of columns given by: batch_start, this_batch_size
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, batch),
                [&](const lno_t j)
                {
                  accum[j] = y(row, batch_start + j);
                });
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, rowSize),
                [&](const lno_t j)
                {
                  lno_t col = rowEntries[j];
                  scalar_t val = rowValues[j];
                  for(lno_t k = 0; k < batch; k++)
                    accum[k] -= val * x(col, batch_start + k);
                });
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, batch),
                [&](const lno_t j)
                {
                  scalar_t newXval = x(row, batch_start + j) * (Kokkos::ArithTraits<scalar_t>::one() - omega);
                  x(row, batch_start + j) = newXval + k * accum[j];
                });
            }
          }
        });
    }

    //offset of each cluster in compressed
    offset_view_t compressedOffsets;
    //output: the compact representation of all clusters
    unit_view_t compressed;
    X_t x;
    Y_t y;
    scalar_t omega;
    lno_t colorSetBegin;
    lno_t colorSetEnd;
  };

  struct PermuteApplyTeam
  {
    PermuteApplyTeam(
        const offset_view_t& compressedOffsets_, const unit_view_t& compressed_,
        const X_t& x_, const Y_t& y_, scalar_t omega_)
      : compressedOffsets(compressedOffsets_), compressed(compressed_),
      x(x_), y(y_), omega(omega_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const team_member_t t) const
    {
      lno_t num_vecs = x.extent(1);
      lno_t teamClusterBegin = colorSetBegin + t.league_rank() * t.team_size();
      lno_t teamClusterEnd = teamClusterBegin + t.team_size();
      if(teamClusterEnd > colorSetEnd)
        teamClusterEnd = colorSetEnd;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, teamClusterEnd - teamClusterBegin),
        [&](const lno_t threadWork)
        {
          lno_t c = teamClusterBegin + threadWork;
          //compressed layout is contiguous, and range policy will just be over the
          //color set, so no need to map the work-item index to a cluster.
          KokkosKernels::Impl::MemStream<unit_t>
            block(&compressed(compressedOffsets(c)));
          //1. get #rows in cluster
          lno_t clusterSize = block.template readSingle<lno_t>();
          //2. get the list of rows in the cluster
          lno_t clusterRowBegin = block.template readSingle<lno_t>();
          //3. store the number of entries in each row (excluding diagonals)
          lno_t* rowSizes = block.template getArray<lno_t>(clusterSize);
          //4. for each row: inverse diagonal, entries and values per row.
          //                 the first row must be scalar-aligned,
          //                 and each values array is also aligned.
          for(lno_t i = 0; i < clusterSize; i++)
          {
            lno_t row = clusterRowBegin + i;
            lno_t rowSize = rowSizes[i];
            //divide up storage for the row:
            //determine where diag^-1, entries, values will go
            comp_scalar_t invDiag = block.template readSingle<comp_scalar_t>();
            lno_t* rowEntries = block.template getArray<lno_t>(rowSize);
            comp_scalar_t* rowValues = block.template getArray<comp_scalar_t>(rowSize);
            constexpr lno_t colBatchSize = 8;
            scalar_t accum[colBatchSize];
            scalar_t k = omega * invDiag;
            for(lno_t batch_start = 0; batch_start < num_vecs; batch_start += colBatchSize)
            {
              lno_t batch = colBatchSize;
              if(batch_start + batch > num_vecs)
                batch = num_vecs - batch_start;
              //the current batch of columns given by: batch_start, this_batch_size
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, batch),
                [&](const lno_t j)
                {
                  accum[j] = y(row, batch_start + j);
                });
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, rowSize),
                [&](const lno_t j)
                {
                  lno_t col = rowEntries[j];
                  scalar_t val = rowValues[j];
                  for(lno_t k = 0; k < batch; k++)
                    accum[k] -= val * x(col, batch_start + k);
                });
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, batch),
                [&](const lno_t j)
                {
                  scalar_t newXval = x(row, batch_start + j) * (Kokkos::ArithTraits<scalar_t>::one() - omega);
                  x(row, batch_start + j) = newXval + k * accum[j];
                });
            }
          }
        });
    }

    //offset of each cluster in compressed
    offset_view_t compressedOffsets;
    //output: the compact representation of all clusters
    unit_view_t compressed;
    X_t x;
    Y_t y;
    scalar_t omega;
    lno_t colorSetBegin;
    lno_t colorSetEnd;
  };
};

}}


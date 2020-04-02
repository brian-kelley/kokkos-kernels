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
}

template<>
struct get_compact_scalar<true, Kokkos::complex<double>>
{
  using type = Kokkos::complex<float>;
}

template <bool compactScalar, typename HandleType, typename rowmap_t, typename entries_t, typename values_t>
struct ClusterCompression
{
  using mem_space = typename HandleType::HandlePersistentMemorySpace;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  using scalar_t = typename values_t::non_const_value_type;
  using comp_scalar_t = typename get_compact_scalar<compactScalar, scalar_t>::type;
  //Compressed representation measures memory in terms of unit_t
  //(the smaller of comp_scalar_t and lno_t)
  //This has the benefit that aligning a unit_t
  //to at least one of lno_t and comp_scalar_t (possibly both, if same size) is a no-op
  using unit_t = std::conditional<
    (sizeof(lno_t) < sizeof(comp_scalar_t)),
    lno_t, comp_scalar_t>;
  using unit_view_t = Kokkos::View<unit_t*, mem_space, Kokkos::MemoryTraits<Kokkos::Aligned>>;
  //Since the sizes of both lno_t and comp_scalar_t should be powers of two,
  //their sizes should be exact multiples of unit_t
  static_assert(sizeof(lno_t) % sizeof(unit_t) == 0,
      "Expect lno_t to have a power-of-two size.");
  static_assert(sizeof(comp_scalar_t) % sizeof(unit_t) == 0,
      "Expect lno_t to have a power-of-two size.");

  //Functor to compute the compressed size of each cluster
  //Also reorders clusters to be grouped by color
  struct CompressedSizeFunctor
  {
    CompressedSizeFunctor(
        const rowmap_t& rowmap_, const entries_t& entries_,
        const entries_t& clusterOffsets_, const entries_t& clusterVerts_,
        const entries_t& storageSize_)
      : rowmap(rowmap_), entries(entries_), clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), storageSize(storageSize_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t dstC) const
    {
      //srcC is the real cluster (corresponding to clusterOffsets and clusterVerts)
      lno_t srcC = colorSets(dstC);
      lno_t clusterBegin = clusterOffsets(srcC);
      lno_t clusterEnd = clusterOffsets(srcC + 1);
      lno_t clusterSize = clusterEnd - clusterBegin;
      constexpr int unitsPerOrdinal = sizeof(lno_t) / sizeof(unit_t);
      constexpr int unitsPerScalar = sizeof(comp_scalar_t) / sizeof(unit_t);
      //lno_t and comp_scalar_t both always have power-of-two sizes,
      //so one size is always a multiple of the other.
      //Simulate the memory layout with a pointer to unit_t.
      unit_t* storagePtr = nullptr;
      //1. store #rows in cluster
      storagePtr += unitsPerOrdinal;
      //2. store the list of rows in the cluster
      storagePtr += clusterSize * unitsPerOrdinal;
      //3. store the number of entries in each row (excluding diagonals)
      storagePtr += clusterSize * unitsPerOrdinal;
      //4. for each row: inverse diagonal, entries and values per row.
      //                 the first row must be scalar-aligned,
      //                 and each values array is also aligned.
      storagePtr = (unit_t*) alignPtr<unit_t*, comp_scalar_t*>(storagePtr);
      for(lno_t i = clusterBegin; i < clusterEnd; i++)
      {
        lno_t row = clusterVerts(i);
        //space for the inverse diagonal
        storagePtr += unitsPerScalar;
        //count off-diagonal entries to store
        int numOffDiag = 0;
        for(size_type j = rowmap(row); j < rowmap(row + 1); j++)
        {
          if(entries(j) != row)
            numOffDiag++;
        }
        //space for entries
        storagePtr = (unit_t*) alignPtr<unit_t*, lno_t*>(storagePtr);
        storagePtr += numOffDiag * unitsPerOrdinal;
        storagePtr = (unit_t*) alignPtr<unit_t*, comp_scalar_t*>(storagePtr);
        storagePtr += numOffDiag * unitsPerScalar;
      }
      //finally, make sure that the storage is aligned to both comp_scalar_t and lno_t
      //(at least one of these will be a no-op)
      storagePtr = (unit_t*) alignPtr<unit_t*, lno_t*>(storagePtr);
      storagePtr = (unit_t*) alignPtr<unit_t*, comp_scalar_t*>(storagePtr);
      storageSize(dstC) = storagePtr - ((unit_t*) nullptr);
      if(dstC == storageSize.extent(0) - 2)
        storageSize(storageSize.extent(0) - 1) = 0;
    }

    rowmap_t rowmap;
    entries_t entries;
    entries_t clusterOffsets;
    entries_t clusterVerts;
    entries_t colorSets;      //aka color_adj
    rowmap_t storageSize;     //output: the number of unit_t needed to store each cluster
  };

  //Functor to compute the compressed representation of clusters
  struct CompressFunctor
  {
    CompressFunctor(
        const rowmap_t& rowmap_, const entries_t& entries_, const values_t& values_,
        const entries_t& clusterOffsets_, const entries_t& clusterVerts_,
        const rowmap_t& compressedOffsets_, const unit_view_t& compressed_)
      : rowmap(rowmap_), entries(entries_), values(values_),
      clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), compressedOffsets(compressedOffsets_), compressed(compressed_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t dstC) const
    {
      //srcC is the real cluster (corresponding to clusterOffsets and clusterVerts)
      lno_t srcC = colorSets(dstC);
      lno_t clusterBegin = clusterOffsets(srcC);
      lno_t clusterEnd = clusterOffsets(srcC + 1);
      lno_t clusterSize = clusterEnd - clusterBegin;
      constexpr int unitsPerOrdinal = sizeof(lno_t) / sizeof(unit_t);
      constexpr int unitsPerScalar = sizeof(comp_scalar_t) / sizeof(unit_t);
      //lno_t and comp_scalar_t both always have power-of-two sizes,
      //so one size is always a multiple of the other.
      //Simulate the memory layout with a pointer to unit_t.
      unit_t* storagePtr = &compressed(compressedOffsets(dstC));
      //1. store #rows in cluster
      *((lno_t*) storagePtr) = clusterSize;
      storagePtr += unitsPerOrdinal;
      //2. store the list of rows in the cluster
      lno_t* clusterRows = (lno_t*) storagePtr;
      for(lno_t i = 0; i < clusterSize; i++)
      {
        clusterRows[i] = clusterVerts(clusterBegin + i);
      }
      storagePtr += clusterSize * unitsPerOrdinal;
      //3. store the number of entries in each row (excluding diagonals)
      lno_t* rowSizes = clusterRows + clusterSize;
      for(lno_t i = 0; i < clusterSize; i++)
      {
        int numOffDiag = 0;
        for(size_type j = rowmap(row); j < rowmap(row + 1); j++)
        {
          if(entries(j) != row)
            numOffDiag++;
        }
        rowSizes[i] = numOffDiag;
      }
      storagePtr += clusterSize * unitsPerOrdinal;
      //4. for each row: inverse diagonal, entries and values per row.
      //                 the first row must be scalar-aligned,
      //                 and each values array is also aligned.
      storagePtr = (unit_t*) alignPtr<unit_t*, comp_scalar_t*>(storagePtr);
      for(lno_t i = 0; i < clusterSize; i++)
      {
        lno_t row = clusterVerts(clusterBegin + i);
        //divide up storage for the row:
        //determine where diag^-1, entries, values will go
        comp_scalar_t* invDiag = (comp_scalar_t*) storagePtr;
        storagePtr += unitsPerScalar;
        storagePtr = (unit_t*) alignPtr<unit_t*, lno_t*>(storagePtr);
        lno_t* rowEntries = (lno_t*) storagePtr;
        storagePtr += rowSizes[i] * unitsPerOrdinal;
        storagePtr = (unit_t*) alignPtr<unit_t*, comp_scalar_t*>(storagePtr);
        comp_scalar_t* rowValues = (comp_scalar_t*) storagePtr;
        storagePtr += rowSizes[i] * unitsPerScalar;
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
    //offset of each cluster in compressed
    rowmap_t compressedOffsets;
    //output: the compact representation of all clusters
    unit_view_t compressed;
  };
}

template <bool compactScalar, typename HandleType, typename rowmap_t, typename entries_t, typename values_t, typename X_t, typename Y_t>
struct CompressedClusterApply
{
  using mem_space = typename HandleType::HandlePersistentMemorySpace;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  using scalar_t = typename values_t::non_const_value_type;
  using comp_scalar_t = typename get_compact_scalar<compactScalar, scalar_t>::type;
  //Compressed representation measures memory in terms of unit_t
  //(the smaller of comp_scalar_t and lno_t)
  //This has the benefit that aligning a unit_t
  //to at least one of lno_t and comp_scalar_t (possibly both, if same size) is a no-op
  using unit_t = std::conditional<
    (sizeof(lno_t) < sizeof(comp_scalar_t)),
    lno_t, comp_scalar_t>;
  using unit_view_t = Kokkos::View<unit_t*, mem_space, Kokkos::MemoryTraits<Kokkos::Aligned>>;

  struct ApplyFunctor
  {
    ApplyFunctor(
        const rowmap_t& compressedOffsets_, const unit_view_t& compressed_,
        const X_t& x_, const Y_t& y_, scalar_t omega_)
      : compressedOffsets(compressedOffsets_), compressed(compressed_), x(x_), y(y_), omega(omega_)
    {}

    KOKKOS_FORCEINLINE_FUNCTION
    void rowApply(comp_scalar_t invDiag, lno_t row, lno_t rowSize, lno_t* rowCols, comp_scalar_t* rowVals) const
    {
      lno_t num_vecs = x.extent(1);
      constexpr colBatchSize = 8;
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
          lno_t col = rowCols[i];
          scalar_t val = rowVals[i];
          for(lno_t i = 0; i < batch; i++)
            accum[i] -= val * x(col, batch_start + i);
        }
        for(lno_t i = 0; i < batch; i++)
        {
          x(row, batch_start + i) *= (Kokkos::ArithTraits<scalar_t>::one() - omega);
          x(row, batch_start + i) += k * sum[i];
        }
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t c) const
    {
      constexpr int unitsPerOrdinal = sizeof(lno_t) / sizeof(unit_t);
      constexpr int unitsPerScalar = sizeof(comp_scalar_t) / sizeof(unit_t);
      //lno_t and comp_scalar_t both always have power-of-two sizes,
      //so one size is always a multiple of the other.
      //Simulate the memory layout with a pointer to unit_t.
      unit_t* storagePtr = &compressed(compressedOffsets(c));
      //1. get #rows in cluster
      lno_t clusterSize = *((lno_t*) storagePtr);
      storagePtr += unitsPerOrdinal;
      //2. get the list of rows in the cluster
      lno_t* clusterRows = (lno_t*) storagePtr;
      storagePtr += clusterSize * unitsPerOrdinal;
      //3. store the number of entries in each row (excluding diagonals)
      lno_t* rowSizes = clusterRows + clusterSize;
      storagePtr += clusterSize * unitsPerOrdinal;
      //4. for each row: inverse diagonal, entries and values per row.
      //                 the first row must be scalar-aligned,
      //                 and each values array is also aligned.
      storagePtr = (unit_t*) alignPtr<unit_t*, comp_scalar_t*>(storagePtr);
      for(lno_t i = 0; i < clusterSize; i++)
      {
        lno_t row = clusterVerts(clusterBegin + i);
        lno_t rowSize = rowSizes[i];
        //divide up storage for the row:
        //determine where diag^-1, entries, values will go
        comp_scalar_t invDiag = *((comp_scalar_t*) storagePtr);
        storagePtr += unitsPerScalar;
        storagePtr = (unit_t*) alignPtr<unit_t*, lno_t*>(storagePtr);
        lno_t* rowEntries = (lno_t*) storagePtr;
        storagePtr += rowSize * unitsPerOrdinal;
        storagePtr = (unit_t*) alignPtr<unit_t*, comp_scalar_t*>(storagePtr);
        comp_scalar_t* rowValues = (comp_scalar_t*) storagePtr;
        storagePtr += rowSize * unitsPerScalar;
        rowApply(invDiag, row, rowSize, rowEntries, rowValues);
      }
    }

    //offset of each cluster in compressed
    rowmap_t compressedOffsets;
    //output: the compact representation of all clusters
    unit_view_t compressed;
    X_t x;
    Y_t y;
    scalar_t omega;
  };
};

}}


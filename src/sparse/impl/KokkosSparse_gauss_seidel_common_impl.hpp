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

#ifndef _KOKKOS_GS_COMMON_IMP_HPP
#define _KOKKOS_GS_COMMON_IMP_HPP

#include "KokkosKernels_Utils.hpp"
#include "Kokkos_ArithTraits.hpp"

namespace KokkosSparse {
namespace Impl {

template<typename lno_t, typename mat_scalar_t, typename vec_scalar_t, typename X_t, typename Y_t, typename team_member_t>
struct GS_RowApply
{
  template<int batchSize>
  KOKKOS_FORCEINLINE_FUNCTION static void
  gsThreadRowApplyBatch(int batchStart, lno_t row, lno_t rowLen, lno_t* rowEntries, mat_scalar_t* rowValues, vec_scalar_t rowXCoef, vec_scalar_t dotCoef, lno_t numVecs, const X_t& x, const Y_t& y, const team_member_t& t)
  {
    using Reducer = KokkosKernels::Impl::array_sum_reduce<vec_scalar_t, batchSize>; 
    Reducer sum;
    Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(t, rowLen),
      [&](lno_t i, Reducer& lsum)
      {
        lno_t colIndex = rowEntries[i];
        //here, converting from mat_scalar_t to vec_scalar_t
        vec_scalar_t val = rowValues[i];
        for(int j = 0; j < batchSize; j++)
          lsum.data[j] += val * x(colIndex, batchStart + j);
      }, sum);
    //Use the fact that ThreadVectorRange reduction results are broadcast
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(t, batchSize),
      [&](lno_t i)
      {
        vec_scalar_t oldX = x(row, batchStart + i);
        x(row, batchStart + i) = rowXCoef * oldX + dotCoef * (y(row, batchStart + i) - sum.data[i]);
      });
  }

  //This is used by both point and cluster GS, to do efficient Gauss-Seidel on a single row, with possibly multiple RHS.
  //rowXCoef depends on whether the row includes the diagonal entry:
  //  if it does (Point GS), then rowXCoef = 1
  //  if it doesn't (Cluster GS), then rowXCoef = 1-omega
  //dotCoef is omega/A[row,row]
  KOKKOS_FORCEINLINE_FUNCTION static void
  gsThreadRowApply(lno_t row, lno_t rowLen, lno_t* rowEntries, mat_scalar_t* rowValues, vec_scalar_t rowXCoef, vec_scalar_t dotCoef, lno_t numVecs, const X_t& x, const Y_t& y, const team_member_t& t)
  {
    //Unroll up to 4 vectors from LHS/RHS at a time. This means that matrix row accesses are optimal for <= 4 vectors.
    for(lno_t batchStart = 0; batchStart < numVecs;)
    {
      switch(numVecs - batchStart)
      {
        #define COL_BATCH_CASE(n) \
          gsThreadRowApplyBatch<n>(batchStart, row, rowLen, rowEntries, rowValues, rowXCoef, dotCoef, numVecs, x, y, t); \
          batchStart += n;
        case 1:
          COL_BATCH_CASE(1);
          break;
        case 2:
          COL_BATCH_CASE(2);
          break;
        case 3:
          COL_BATCH_CASE(3);
          break;
        default:
          COL_BATCH_CASE(4);
      }
    }
  }
};

}}  //namespace KokkosSparse::Impl

#endif


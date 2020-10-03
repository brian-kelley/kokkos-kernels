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

#ifndef _KOKKOSGRAPH_MATCHING_HPP
#define _KOKKOSGRAPH_MATCHING_HPP

#include "Kokkos_Core.hpp"
#include "KokkosKernels_Utils.hpp"
#include <cstdint>

namespace KokkosGraph {
namespace Experimental {
namespace Impl {

template<typename device_t, typename rowmap_t, typename entries_t, typename lno_view_t>
struct MaximalMatching
{
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  //The type of status/priority values.
  using status_t = uint64_t;
  using status_view_t = Kokkos::View<status_t*, mem_space>;
  using range_pol = Kokkos::RangePolicy<exec_space>;
  using team_pol = Kokkos::TeamPolicy<exec_space>;
  using team_mem = typename team_pol::member_type;
  using all_worklists_t = Kokkos::View<lno_t**, Kokkos::LayoutLeft, mem_space>;
  using worklist_t = Kokkos::View<lno_t*, Kokkos::LayoutLeft, mem_space>;

  KOKKOS_INLINE_FUNCTION static uint32_t xorshift32(uint32_t in)
  {
    uint32_t x = in;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
  }

  KOKKOS_INLINE_FUNCTION static uint64_t xorshift64(uint64_t in)
  {
    uint64_t x = in;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
  }

  // Priority values 0 and max are special, they mean the vertex is
  // in the independent set or eliminated from consideration, respectively.
  // Values in between represent a priority for being added to the set,
  // based on degree and vertex ID as a tiebreak
  //   (higher priority = less preferred to being in the independent set)

  static constexpr status_t IN_SET = 0;
  static constexpr status_t OUT_SET = ~IN_SET;

  MaximalMatching(const rowmap_t& rowmap_, const entries_t& entries_)
    : rowmap(rowmap_), entries(entries_), numVerts(rowmap.extent(0) - 1)
  {
    //Find #bits to represent maximum value possible for a unique edge ID.
    status_t i = (status_t) (numVerts - 2) * (numVerts - 1) + 1;
    int idBits = 0;
    while(i)
    {
      i >>= 1;
      idBits++;
    }
    //Compute hash mask for the upper bits, which won't interfere with the edge ID
    hashMask = 1;
    hashMask <<= idBits;
    hashMask--;
    hashMask = ~hashMask;
    vertStatus = status_view_t(Kokkos::ViewAllocateWithoutInitializing("VertStatus"), numVerts);
    allWorklists = Kokkos::View<lno_t**, Kokkos::LayoutLeft, mem_space>(Kokkos::ViewAllocateWithoutInitializing("AllWorklists"), numVerts, 2);
    matches = lno_view_t(Kokkos::ViewAllocateWithoutInitializing("Matches"), numVerts);
  }

  static KOKKOS_FORCEINLINE_FUNCTION status_t computeEdgeStatus(status_t v1, status_t v2, status_t hashedRound, status_t hashMask, status_t numVerts)
  {
    if(v1 > v2)
    {
      status_t temp = v1;
      v1 = v2;
      v2 = temp;
    }
    //Overall status requirements:
    //  -Most significant bits should be pseudorandom
    //  -No collisions should be possible between any two edges.
    //  -So, the least significant bits will hold v1 * numVerts + v2 + 1.
    return (xorshift64(v1 ^ xorshift64(v2 ^ hashedRound)) & hashMask) | (v1 * numVerts + v2 + 1);
  }

  struct RefreshVertexStatus
  {
    RefreshVertexStatus(const status_view_t& vertStatus_, const worklist_t& worklist_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_, status_t hashedRound_, status_t hashMask_, bool firstRound_)
      : vertStatus(vertStatus_), worklist(worklist_), rowmap(rowmap_), entries(entries_), nv(nv_), hashedRound(hashedRound_), hashMask(hashMask_), firstRound(firstRound_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      status_t minStat = OUT_SET;
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      if(!firstRound && vertStatus(i) == OUT_SET)
        return;
      for(size_type j = rowBegin; j < rowEnd; j++)
      {
        lno_t nei = entries(j);
        if(nei == i || nei >= nv)
          continue;
        //If nei is OUT_SET, then so is the edge to it.
        //The result is to not change the minimum for i.
        if(vertStatus(nei) != OUT_SET)
        {
          status_t edgeStatus = computeEdgeStatus(i, nei, hashedRound, hashMask, nv);
          if(edgeStatus < minStat)
            minStat = edgeStatus;
        }
      }
      //In the first round, overwrite every status.
      //In later rounds, never overwrite a value of OUT_SET with a lower value.
      vertStatus(i) = minStat;
    }

    status_view_t vertStatus;
    worklist_t worklist;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
    status_t hashedRound;
    status_t hashMask;
    bool firstRound;
  };

  struct DecideMatchesFunctor
  {
    DecideMatchesFunctor(const status_view_t& vertStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_, const worklist_t& worklist_, const lno_view_t& matches_, status_t hashedRound_, status_t hashMask_)
      : vertStatus(vertStatus_), rowmap(rowmap_), entries(entries_), nv(nv_), worklist(worklist_), matches(matches_), hashedRound(hashedRound_), hashMask(hashMask_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      //Processing row i.
      //Iterate over each edge. If nei > i, compute the edge status and check if it's the minimum for both endpoints.
      //If it is, match nei with i, then mark 
      status_t iStat = vertStatus(i);
      //s is the status which must be the minimum among all neighbors
      //to decide that i is IN_SET.
      size_type rowBegin = rowmap(i); size_type rowEnd = rowmap(i + 1);
      lno_t mergeNei = i;
      for(size_type j = rowBegin; j < rowEnd; j++)
      {
        lno_t nei = entries(j);
        //Only want to identify edges once in this loop, so only consider (i, nei) with i<nei
        if(nei <= i || nei >= nv)
          continue;
        status_t neiStat = vertStatus(nei);
        status_t edgeStatus = computeEdgeStatus(i, nei, hashedRound, hashMask, nv);
        if(edgeStatus == iStat && edgeStatus == neiStat)
        {
          mergeNei = nei;
          break;
        }
      }
      if(mergeNei != i)
      {
        //Merge the edge. Mark endpoints as OUT_SET.
        //This means that any edges incident to i or mergeNei will also have status OUT_SET.
        matches(i) = i;
        matches(mergeNei) = i;
        vertStatus(i) = OUT_SET;
        vertStatus(mergeNei) = OUT_SET;
      }
    }

    status_view_t vertStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
    status_t hashedRound;
    status_t hashMask;
    worklist_t worklist;
    lno_view_t matches;
  };

  struct InitWorklistFunctor
  {
    InitWorklistFunctor(const worklist_t& worklist_)
      : worklist(worklist_)
    {}
    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      worklist(i) = i;
    }
    worklist_t worklist;
  };

  struct CompactWorklistFunctor
  {
    CompactWorklistFunctor(const worklist_t& src_, const worklist_t& dst_, const status_view_t& status_)
      : src(src_), dst(dst_), status(status_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w, lno_t& lNumInSet, bool finalPass) const
    {
      lno_t i = src(w);
      status_t s = status(i);
      if(s != OUT_SET)
      {
        //next worklist needs to contain i
        if(finalPass)
          dst(lNumInSet) = i;
        lNumInSet++;
      }
    }

    worklist_t src;
    worklist_t dst;
    status_view_t status;
  };

  lno_view_t compute()
  {
    //Initialize first worklist to 0...numVerts
    worklist_t vertWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 0);
    worklist_t tempWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 1);
    Kokkos::parallel_for(range_pol(0, numVerts), InitWorklistFunctor(vertWorklist));
    //Also init the matches: start with every vertex unmatched
    Kokkos::parallel_for(range_pol(0, numVerts), InitWorklistFunctor(matches));
    status_t round = 0;
    lno_t workLen = numVerts;
    while(true)
    {
      //Compute new vertex statuses
      status_t hashedRound = xorshift64(round);
      Kokkos::parallel_for(range_pol(0, workLen), RefreshVertexStatus(vertStatus, vertWorklist, rowmap, entries, numVerts, hashedRound, hashMask, round == 0));
      //Then find matches
      Kokkos::parallel_for(range_pol(0, workLen), DecideMatchesFunctor(vertStatus, rowmap, entries, numVerts, vertWorklist, matches, hashedRound, hashMask));
      //Compact worklist (keep vertices which are not OUT_SET)
      Kokkos::parallel_scan(range_pol(0, workLen), CompactWorklistFunctor(vertWorklist, tempWorklist, vertStatus), workLen);
      if(workLen == 0)
        break;
      std::swap(vertWorklist, tempWorklist);
      round++;
      if(round == 1000)
        break;
    }
    return matches;
  }

  rowmap_t rowmap;
  entries_t entries;
  lno_t numVerts;
  status_view_t vertStatus;
  all_worklists_t allWorklists;
  lno_view_t matches;
  status_t hashMask;
};

}}}

#endif

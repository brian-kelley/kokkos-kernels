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
  using status_t = size_type;
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
    status_t i = (status_t) numVerts * numVerts + 1;
    nvBits = 0;
    while(i)
    {
      i >>= 1;
      nvBits++;
    }
    vertStatus = status_view_t(Kokkos::ViewAllocateWithoutInitializing("VertStatus"), numVerts);
    allWorklists = Kokkos::View<lno_t**, Kokkos::LayoutLeft, mem_space>(Kokkos::ViewAllocateWithoutInitializing("AllWorklists"), numVerts, 2);
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
    RefreshVertexStatus(const status_view_t& vertStatus_, const worklist_t& worklist_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_, status_t hashMask_)
      : colStatus(colStatus_), worklist(worklist_), rowStatus(rowStatus_), rowmap(rowmap_), entries(entries_), nv(nv_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      size_type minStat = OUT_SET;
      for(size_type j = rowBegin; j < rowEnd; j++)
      {
        lno_t nei = entries(j);
        if(nei == i || nei >= nv)
          continue;
        status_t edgeStatus = computeEdgeStatus(i, nei, hashedRound, hashMask, nv);
        if(edgeStatus < minStat)
          minStat = edgeStatus;
      }
      if(minStat == IN_SET)
        minStat = OUT_SET;
      vertStatus(i) = minStat;
    }

    status_view_t vertStatus;
    worklist_t worklist;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
    status_t hashedRound;
    status_t hashMask;
  };

  struct DecideMatchesFunctor
  {
    DecideMatchesFunctor(const status_view_t& rowStatus_, const status_view_t& colStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_, const worklist_t& worklist_, lno_t worklistLen_)
      : rowStatus(rowStatus_), colStatus(colStatus_), rowmap(rowmap_), entries(entries_), nv(nv_), worklist(worklist_), worklistLen(worklistLen_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      //Processing row i.
      //Iterate over each edge. If nei > i, compute the edge status and check if it's the minimum for both endpoints.
      //If it is, match nei with i, then mark 
      status_t s = rowStatus(i);
      if(s == IN_SET || s == OUT_SET)
        return;
      //s is the status which must be the minimum among all neighbors
      //to decide that i is IN_SET.
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      bool neiOut = false;
      bool neiMismatchS = false;
      for(size_type j = rowBegin; j <= rowEnd; j++)
      {
        lno_t nei = (j == rowEnd) ? i : entries(j);
        if(nei >= nv)
          continue;
        status_t neiStat = colStatus(nei);
        if(neiStat == OUT_SET)
        {
          neiOut = true;
          break;
        }
        else if(neiStat != s)
        {
          neiMismatchS = true;
        }
      }
      if(neiOut)
      {
        //In order to make future progress, need to update the
        //col statuses for all neighbors of i.
        rowStatus(i) = OUT_SET;
      }
      else if(!neiMismatchS)
      {
        //all neighboring col statuses match s, therefore s is the minimum status among all d2 neighbors
        rowStatus(i) = IN_SET;
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const team_mem& t) const
    {
      using OrReducer = Kokkos::BOr<int>;
      lno_t w = t.league_rank() * t.team_size() + t.team_rank();
      if(w >= worklistLen)
        return;
      lno_t i = worklist(w);
      //Processing row i.
      status_t s = rowStatus(i);
      if(s == IN_SET || s == OUT_SET)
        return;
      //s is the status which must be the minimum among all neighbors
      //to decide that i is IN_SET.
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      lno_t rowLen = rowEnd - rowBegin;
      int flags = 0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, rowLen + 1),
      [&](lno_t j, int& lflags)
      {
        lno_t nei = (j == rowLen) ? i : entries(rowBegin + j);
        if(nei >= nv)
          return;
        status_t neiStat = colStatus(nei);
        if(neiStat == OUT_SET)
          lflags |= NEI_OUT_SET;
        else if(neiStat != s)
          lflags |= NEI_DIFFERENT_STATUS;
      }, OrReducer(flags));
      Kokkos::single(Kokkos::PerThread(t),
      [&]()
      {
        if(flags & NEI_OUT_SET)
        {
          //In order to make future progress, need to update the
          //col statuses for all neighbors of i.
          rowStatus(i) = OUT_SET;
        }
        else if(!(flags & NEI_DIFFERENT_STATUS))
        {
          //all neighboring col statuses match s, therefore s is the minimum status among all d2 neighbors
          rowStatus(i) = IN_SET;
        }
      });
    }

    status_view_t rowStatus;
    status_view_t colStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
    worklist_t worklist;
    lno_t worklistLen;
  };

  struct CountInSet
  {
    CountInSet(const status_view_t& rowStatus_)
      : rowStatus(rowStatus_)
    {}
    KOKKOS_INLINE_FUNCTION void operator()(lno_t i, lno_t& lNumInSet) const
    {
      if(rowStatus(i) == IN_SET)
        lNumInSet++;
    }
    status_view_t rowStatus;
  };

  struct CompactInSet
  {
    CompactInSet(const status_view_t& rowStatus_, const lno_view_t& setList_)
      : rowStatus(rowStatus_), setList(setList_)
    {}
    KOKKOS_INLINE_FUNCTION void operator()(lno_t i, lno_t& lNumInSet, bool finalPass) const
    {
      if(rowStatus(i) == IN_SET)
      {
        if(finalPass)
          setList(lNumInSet) = i;
        lNumInSet++;
      }
    }
    status_view_t rowStatus;
    lno_view_t setList;
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
      if(s != IN_SET && s != OUT_SET)
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
    worklist_t rowWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 0);
    Kokkos::parallel_for(range_pol(0, numVerts), InitWorklistFunctor(rowWorklist));
    worklist_t colWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 1);
    Kokkos::parallel_for(range_pol(0, numVerts), InitWorklistFunctor(colWorklist));
    worklist_t thirdWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 2);
    auto execSpaceEnum = KokkosKernels::Impl::kk_get_exec_space_type<exec_space>();
    bool useTeams = (execSpaceEnum == KokkosKernels::Impl::Exec_CUDA) && (entries.extent(0) / numVerts >= 16);
    int vectorLength = KokkosKernels::Impl::kk_get_suggested_vector_size(numVerts, entries.extent(0), execSpaceEnum);
    int round = 0;
    lno_t rowWorkLen = numVerts;
    lno_t colWorkLen = numVerts;
    int refreshColTeamSize = 0;
    int decideSetTeamSize = 0;
    if(useTeams)
    {
      team_pol dummyPolicy(1, 1, vectorLength);
      //Compute the recommended team size for RefreshColStatus and DecideSetFunctor (will be constant)
      {
        RefreshColStatus refreshCol(colStatus, colWorklist, rowStatus, rowmap, entries, numVerts, colWorkLen);
        refreshColTeamSize = dummyPolicy.team_size_max(refreshCol, Kokkos::ParallelForTag());
      }
      {
        DecideSetFunctor decideSet(rowStatus, colStatus, rowmap, entries, numVerts, rowWorklist, rowWorkLen);
        decideSetTeamSize = dummyPolicy.team_size_max(decideSet, Kokkos::ParallelForTag());
      }
    }
    while(true)
    {
      //Compute new row statuses
      Kokkos::parallel_for(range_pol(0, rowWorkLen), RefreshRowStatus(rowStatus, rowWorklist, nvBits, round));
      //Compute new col statuses
      {
        RefreshColStatus refreshCol(colStatus, colWorklist, rowStatus, rowmap, entries, numVerts, colWorkLen);
        if(useTeams)
          Kokkos::parallel_for(team_pol((colWorkLen + refreshColTeamSize - 1) / refreshColTeamSize, refreshColTeamSize, vectorLength), refreshCol);
        else
          Kokkos::parallel_for(range_pol(0, colWorkLen), refreshCol);
      }
      //Decide row statuses where enough information is available
      {
        DecideSetFunctor decideSet(rowStatus, colStatus, rowmap, entries, numVerts, rowWorklist, rowWorkLen);
        if(useTeams)
          Kokkos::parallel_for(team_pol((rowWorkLen + decideSetTeamSize - 1) / decideSetTeamSize, decideSetTeamSize, vectorLength), decideSet);
        else
          Kokkos::parallel_for(range_pol(0, rowWorkLen), decideSet);
      }
      //Compact row worklist
      Kokkos::parallel_scan(range_pol(0, rowWorkLen), CompactWorklistFunctor(rowWorklist, thirdWorklist, rowStatus), rowWorkLen);
      if(rowWorkLen == 0)
        break;
      std::swap(rowWorklist, thirdWorklist);
      //Compact col worklist
      Kokkos::parallel_scan(range_pol(0, colWorkLen), CompactWorklistFunctor(colWorklist, thirdWorklist, colStatus), colWorkLen);
      std::swap(colWorklist, thirdWorklist);
      round++;
    }
    //now that every vertex has been decided IN_SET/OUT_SET,
    //build a compact list of the vertices which are IN_SET.
    lno_t numInSet = 0;
    Kokkos::parallel_reduce(range_pol(0, numVerts), CountInSet(rowStatus), numInSet);
    lno_view_t setList(Kokkos::ViewAllocateWithoutInitializing("D2MIS"), numInSet);
    Kokkos::parallel_scan(range_pol(0, numVerts), CompactInSet(rowStatus, setList));
    return setList;
  }

  rowmap_t rowmap;
  entries_t entries;
  lno_t numVerts;
  status_view_t rowStatus;
  status_view_t colStatus;
  all_worklists_t allWorklists;
  //The number of bits required to represent vertex IDs, in the ECL-MIS tiebreak scheme:
  //  ceil(log_2(numVerts + 1))
  int nvBits;
};

}}}

#endif

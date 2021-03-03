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

#ifndef _KOKKOSGRAPH_MATCHING_IMPL_HPP
#define _KOKKOSGRAPH_MATCHING_IMPL_HPP

#include "Kokkos_Core.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosGraph_ExplicitCoarsening.hpp"
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
  using bool_view_t = Kokkos::View<int8_t*, mem_space>;
  /*
  using bitset_t = Kokkos::Bitset<device_t>;
  using const_bitset_t = Kokkos::ConstBitset<device_t>;
  */

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

  static KOKKOS_FORCEINLINE_FUNCTION status_t computeEdgeStatus(status_t v1, status_t v2, status_t hashedRound, status_t hashMask_, status_t numVerts_, status_t seed_)
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
    return (xorshift64(v1 ^ xorshift64(v2 ^ hashedRound ^ xorshift64(seed_))) & hashMask_) | (v1 * numVerts_ + v2 + 1);
  }

  struct RefreshVertexStatus
  {
    RefreshVertexStatus(const status_view_t& vertStatus_, const worklist_t& worklist_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_, status_t hashedRound_, status_t hashMask_, bool firstRound_, const bool_view_t& isMatched_, lno_t workLen_, status_t seed_)
      : vertStatus(vertStatus_), worklist(worklist_), rowmap(rowmap_), entries(entries_), nv(nv_), hashedRound(hashedRound_), hashMask(hashMask_), firstRound(firstRound_), isMatched(isMatched_), workLen(workLen_), seed(seed_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      status_t minStat = OUT_SET;
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      for(size_type j = rowBegin; j < rowEnd; j++)
      {
        lno_t nei = entries(j);
        if(nei == i || nei >= nv)
          continue;
        //If nei is OUT_SET, then so is the edge to it.
        //The result is to not change the minimum for i.
        if(!isMatched(nei))
        {
          status_t edgeStatus = computeEdgeStatus(i, nei, hashedRound, hashMask, nv, seed);
          if(edgeStatus < minStat)
            minStat = edgeStatus;
        }
      }
      //In the first round, overwrite every status.
      //In later rounds, never overwrite a value of OUT_SET with a lower value.
      vertStatus(i) = minStat;
    }

    KOKKOS_INLINE_FUNCTION void operator()(const team_mem& t) const
    {
      using MinReducer = Kokkos::Min<status_t>;
      lno_t w = t.league_rank() * t.team_size() + t.team_rank();
      if(w >= workLen)
        return;
      lno_t i = worklist(w);
      status_t minStat;
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, rowBegin, rowEnd),
      [&](lno_t j, status_t& ls)
      {
        lno_t nei = entries(j);
        if(nei == i || nei >= nv)
          return;
        //If nei is OUT_SET, then so is the edge to it.
        //The result is to not change the minimum for i.
        if(!isMatched(nei))
        {
          status_t edgeStatus = computeEdgeStatus(i, nei, hashedRound, hashMask, nv, seed);
          if(edgeStatus < ls)
            ls = edgeStatus;
        }
      }, MinReducer(minStat));
      Kokkos::single(Kokkos::PerThread(t),
      [&]()
      {
        vertStatus(i) = minStat;
      });
    }

    status_view_t vertStatus;
    worklist_t worklist;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
    status_t hashedRound;
    status_t hashMask;
    bool firstRound;
    bool_view_t isMatched;
    lno_t workLen;
    status_t seed;
  };

  struct DecideMatchesFunctor
  {
    DecideMatchesFunctor(const status_view_t& vertStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_, const worklist_t& worklist_, const lno_view_t& matches_, status_t hashedRound_, status_t hashMask_, const bool_view_t& isMatched_, lno_t workLen_, status_t seed_)
      : vertStatus(vertStatus_), rowmap(rowmap_), entries(entries_), nv(nv_), worklist(worklist_), matches(matches_), hashedRound(hashedRound_), hashMask(hashMask_), isMatched(isMatched_), workLen(workLen_), seed(seed_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      //Processing row i.
      //Iterate over each edge. If nei > i, compute the edge status and check if it's the minimum for both endpoints.
      //If it is, can match i and nei
      status_t iStat = vertStatus(i);
      if(iStat == OUT_SET)
      {
        //No edges to unmatched neighbors, can't match
        isMatched(i) = 1;
        return;
      }
      //s is the status which must be the minimum among all neighbors
      //to decide that <i,nei> is in the matching.
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      lno_t mergeNei = i;
      for(size_type j = rowBegin; j < rowEnd; j++)
      {
        lno_t nei = entries(j);
        //Only want to identify edges once in this loop, so only consider (i, nei) with i<nei
        if(nei <= i || nei >= nv)
          continue;
        status_t neiStat = vertStatus(nei);
        status_t edgeStatus = computeEdgeStatus(i, nei, hashedRound, hashMask, nv, seed);
        if(edgeStatus == iStat && edgeStatus == neiStat)
        {
          mergeNei = nei;
          break;
        }
      }
      if(mergeNei != i)
      {
        //Found a match. Mark both endpoints as matched.
        matches(mergeNei) = i;
        isMatched(i) = 1;
        isMatched(mergeNei) = 1;
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const team_mem& t) const
    {
      using MinReducer = Kokkos::Min<lno_t>;
      lno_t w = t.league_rank() * t.team_size() + t.team_rank();
      if(w >= workLen)
        return;
      lno_t i = worklist(w);
      //Processing row i.
      //Iterate over each edge. If nei > i, compute the edge status and check if it's the minimum for both endpoints.
      //If it is, can match i and nei
      status_t iStat = vertStatus(i);
      if(iStat == OUT_SET)
      {
        //No edges to unmatched neighbors, can't match
        Kokkos::single(Kokkos::PerThread(t),
        [&]()
        {
          isMatched(i) = 1;
        });
        return;
      }
      //s is the status which must be the minimum among all neighbors
      //to decide that <i,nei> is in the matching.
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      lno_t mergeNei;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, rowBegin, rowEnd),
      [&](size_type j, lno_t& lmatch)
      {
        lno_t nei = entries(j);
        //Don't want to check an edge more than once per round, so only consider (i, nei) with i<nei
        if(nei <= i || nei >= nv)
          return;
        status_t neiStat = vertStatus(nei);
        status_t edgeStatus = computeEdgeStatus(i, nei, hashedRound, hashMask, nv, seed);
        //this is a min-reduction and a valid match for i is unique
        if(edgeStatus == iStat && edgeStatus == neiStat)
          lmatch = nei;
      }, MinReducer(mergeNei));
      if(mergeNei < nv)
      {
        Kokkos::single(Kokkos::PerThread(t),
        [&]()
        {
          //Found a match. Mark both endpoints as matched.
          matches(mergeNei) = i;
          isMatched(i) = 1;
          isMatched(mergeNei) = 1;
        });
      }
    }

    status_view_t vertStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
    worklist_t worklist;
    lno_view_t matches;
    status_t hashedRound;
    status_t hashMask;
    bool_view_t isMatched;
    lno_t workLen;
    status_t seed;
  };

  struct CompactWorklistFunctor
  {
    CompactWorklistFunctor(const worklist_t& src_, const worklist_t& dst_, const bool_view_t& isMatched_, const status_view_t& vertStatus_)
      : src(src_), dst(dst_), isMatched(isMatched_), vertStatus(vertStatus_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w, lno_t& lNumInSet, bool finalPass) const
    {
      lno_t i = src(w);
      if(!isMatched(i))
      {
        //next worklist needs to contain i
        if(finalPass)
        {
          dst(lNumInSet) = i;
        }
        lNumInSet++;
      }
      else if(finalPass)
      {
        //i was matched or marked as singleton last round, so update its status
        vertStatus(i) = OUT_SET;
      }
    }

    worklist_t src;
    worklist_t dst;
    bool_view_t isMatched;
    status_view_t vertStatus;
  };

  void compute()
  {
    auto execSpaceEnum = KokkosKernels::Impl::kk_get_exec_space_type<exec_space>();
    bool useTeams = (execSpaceEnum == KokkosKernels::Impl::Exec_CUDA) && (entries.extent(0) / numVerts >= 16);
    worklist_t vertWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 0);
    worklist_t tempWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 1);
    bool_view_t isMatched("isMatched", numVerts);
    int vectorLength = KokkosKernels::Impl::kk_get_suggested_vector_size(numVerts, entries.extent(0), execSpaceEnum);
    int refreshTeamSize = 0;
    int decideTeamSize = 0;
    if(useTeams)
    {
      team_pol dummyPolicy(1, 1, vectorLength);
      //Compute the recommended team size for RefreshColStatus and DecideSetFunctor (will be constant)
      {
        RefreshVertexStatus refreshVert(vertStatus, vertWorklist, rowmap, entries, numVerts, 0, hashMask, false, isMatched, 0, seed);
        refreshTeamSize = dummyPolicy.team_size_max(refreshVert, Kokkos::ParallelForTag());
      }
      {
        DecideMatchesFunctor decideMatches(vertStatus, rowmap, entries, numVerts, vertWorklist, matches, 0, hashMask, isMatched, 0, seed);
        decideTeamSize = dummyPolicy.team_size_max(decideMatches, Kokkos::ParallelForTag());
      }
    }
    //Initialize first worklist to 0...numVerts
    KokkosKernels::Impl::sequential_fill(vertWorklist);
    //Also init the matches: start with every vertex unmatched (represented as being matched with itself)
    KokkosKernels::Impl::sequential_fill(matches);
    status_t round = 0;
    lno_t workLen = numVerts;
    //while(true)
    {
      //Compute new vertex statuses
      status_t hashedRound = xorshift64(round);
      {
        RefreshVertexStatus refreshVert(vertStatus, vertWorklist, rowmap, entries, numVerts, hashedRound, hashMask, round == 0, isMatched, workLen, seed);
        if(useTeams)
          Kokkos::parallel_for(team_pol((workLen + refreshTeamSize - 1) / refreshTeamSize, refreshTeamSize, vectorLength), refreshVert);
        else
          Kokkos::parallel_for(range_pol(0, workLen), refreshVert);
      }
      //Then find matches
      {
        DecideMatchesFunctor decideMatches(vertStatus, rowmap, entries, numVerts, vertWorklist, matches, hashedRound, hashMask, isMatched, workLen, seed);
        if(useTeams)
          Kokkos::parallel_for(team_pol((workLen + decideTeamSize - 1) / decideTeamSize, decideTeamSize, vectorLength), decideMatches);
        else
          Kokkos::parallel_for(range_pol(0, workLen), decideMatches);
      }
      //Compact worklist (keep vertices which are not OUT_SET)
      //Kokkos::parallel_scan(range_pol(0, workLen), CompactWorklistFunctor(vertWorklist, tempWorklist, isMatched, vertStatus), workLen);
      //if(workLen == 0)
      //  break;
      //std::swap(vertWorklist, tempWorklist);
      //round++;
    }
  }

  rowmap_t rowmap;
  entries_t entries;
  lno_t numVerts;
  status_view_t vertStatus;
  all_worklists_t allWorklists;
  lno_view_t matches;
  status_t hashMask;
  status_t seed;
};

template<typename device_t, typename rowmap_t, typename entries_t, typename lno_view_t>
struct MaximalMatchCoarsening
{
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  using unmanaged_rowmap_t = Kokkos::View<size_type*, mem_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using unmanaged_entries_t = Kokkos::View<lno_t*, mem_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using owned_rowmap_t = Kokkos::View<size_type*, mem_space>;
  using owned_entries_t = Kokkos::View<lno_t*, mem_space>;
  //The type of status/priority values.
  using status_t = uint64_t;
  using status_view_t = Kokkos::View<status_t*, mem_space>;
  using range_pol = Kokkos::RangePolicy<exec_space>;
  using team_pol = Kokkos::TeamPolicy<exec_space>;
  using team_mem = typename team_pol::member_type;
  using all_worklists_t = Kokkos::View<lno_t**, Kokkos::LayoutLeft, mem_space>;
  using worklist_t = Kokkos::View<lno_t*, Kokkos::LayoutLeft, mem_space>;
  using bool_view_t = Kokkos::View<int8_t*, mem_space>;

  // Priority values 0 and max are special, they mean the vertex is
  // in the independent set or eliminated from consideration, respectively.
  // Values in between represent a priority for being added to the set,
  // based on degree and vertex ID as a tiebreak
  //   (higher priority = less preferred to being in the independent set)

  static constexpr status_t IN_SET = 0;
  static constexpr status_t OUT_SET = ~IN_SET;

  MaximalMatchCoarsening(const rowmap_t& rowmap_, const entries_t& entries_)
    : rowmap(rowmap_), entries(entries_), numVerts(rowmap.extent(0) - 1)
  {}

  struct FindRootsFunctor
  {
    FindRootsFunctor(const lno_view_t& matches_, const lno_view_t& labels_)
      : matches(matches_), labels(labels_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i, lno_t& lindex, bool finalPass) const
    {
      if(matches(i) == i)
      {
        //i is a root
        if(finalPass)
          labels(i) = lindex;
        lindex++;
      }
    }

    lno_view_t matches;
    lno_view_t labels;
  };

  struct AssignNonRootsFunctor
  {
    AssignNonRootsFunctor(const lno_view_t& matches_, const lno_view_t& labels_)
      : matches(matches_), labels(labels_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      lno_t match = matches(i);
      if(match != i)
      {
        //i is not a root, so its label is the label of its match
        labels(i) = labels(match);
      }
    }

    lno_view_t matches;
    lno_view_t labels;
  };

  struct PropagateLabelsFunctor
  {
    PropagateLabelsFunctor(const lno_view_t& fineLabels_, const lno_view_t& coarseLabels_)
      : fineLabels(fineLabels_), coarseLabels(coarseLabels_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      fineLabels(i) = coarseLabels(fineLabels(i));
    }

    lno_view_t fineLabels;
    lno_view_t coarseLabels;
  };

  lno_view_t compute(int numSteps, lno_t& numClusters)
  {
    unmanaged_rowmap_t g_rowmap = rowmap;
    unmanaged_entries_t g_entries = entries;
    lno_t g_nv = numVerts;
    owned_rowmap_t temp_rowmap;
    owned_entries_t temp_entries;
    lno_view_t finalLabels;
    for(int step = 0; step < numSteps; step++)
    {
      //Run matching on g
      MaximalMatching<device_t, unmanaged_rowmap_t, unmanaged_entries_t, lno_view_t> matching(g_rowmap, g_entries);
      lno_view_t matches = matching.compute();
      lno_view_t labels(Kokkos::ViewAllocateWithoutInitializing("Labels"), g_nv);
      lno_t coarse_nv;
      Kokkos::parallel_scan(range_pol(0, g_nv), FindRootsFunctor(matches, labels), coarse_nv);
      Kokkos::parallel_for(range_pol(0, g_nv), AssignNonRootsFunctor(matches, labels));
      //First, propagate labels to the finest level
      if(step == 0)
      {
        finalLabels = labels;
      }
      else
      {
        Kokkos::parallel_for(range_pol(0, numVerts), PropagateLabelsFunctor(finalLabels, labels));
      }
      //Then finalize or coarsen the graph
      if(step == numSteps - 1)
      {
        numClusters = coarse_nv;
      }
      else
      {
        //Generate the next g, one level coarser
        owned_rowmap_t next_rowmap;
        owned_entries_t next_entries;
        graph_explicit_coarsen<device_t, unmanaged_rowmap_t, unmanaged_entries_t, lno_view_t, owned_rowmap_t, owned_entries_t>
          (g_rowmap, g_entries, labels, coarse_nv, next_rowmap, next_entries, false);
        //Replace g (setting temp_rowmap/temp_entries also so that they don't get deallocated)
        temp_rowmap = next_rowmap;
        temp_entries = next_entries;
        g_rowmap = temp_rowmap;
        g_entries = temp_entries;
        g_nv = coarse_nv;
      }
    }
    return finalLabels;
  }

  rowmap_t rowmap;
  entries_t entries;
  lno_t numVerts;
  status_view_t vertStatus;
  all_worklists_t allWorklists;
  lno_view_t labels;
  status_t hashMask;
};

}}}

#endif

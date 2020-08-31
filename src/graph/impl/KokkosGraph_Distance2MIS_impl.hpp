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

#ifndef _KOKKOSGRAPH_DISTANCE2_MIS_IMPL_HPP
#define _KOKKOSGRAPH_DISTANCE2_MIS_IMPL_HPP

#include "Kokkos_Core.hpp"
#include "Kokkos_Bitset.hpp"
#include "KokkosKernels_SparseUtils.hpp"
#include <cstdint>

namespace KokkosGraph {
namespace Experimental {
namespace Impl {

/*
 *  100% asynchronous algorithm ideas:
 *    -For each row in team worklist:
 *      -Determine if any neighboring columns are OUT_SET, as well as whether all col statuses match my row status exactly
 *      -If any neighbors are OUT_SET:
 *        -Mark row permanently as OUT_SET.
 *        -Mark all neighboring columns for status update, since their minimum status may now have increased.
 *      -If all neighbor statuses match this row's status, mark this row permanently as IN_SET. Then mark all neighboring columns as OUT_SET.
 *    -Process all pending column updates (atomic_maxing the status with new one, if multiple threads may get the same column)
 *
 *    -Invariants:
 *      -Row status changes exactly once (to either IN_SET or OUT_SET). After this, it never needs to be proccessed again.
 *      -Col status can change multiple times, but it can only increase (up to OUT_SET)
 *        -Therefore, when a column is updated, it converges to the true minimum status over rows
 *
 *    What if a row R 2 hops away becomes IN_SET, and this row doesn't observe the columns changing to OUT_SET?
 *      -It's OK, since at no time can this row observe a mutual neighbor exactly matching its status. It will match R's status, and then it will be OUT_SET).
 *    What if a column's updated status is based on out of date information?
 *      -The minimum is computed as: any are IN_SET? OUT_SET : min(neighbors)
 *      -This quantity may only increase, since rows can only change to IN_SET or OUT_SET, and in either case it increases
 *      -So it's OK, since if it's out of date, it can only be _lower_ than it should be, never allowing a vertex to become IN_SET that shouldn't
 *
 *  Problem still to solve: with priorities chosen only once, will still converge slowly. Need a way to have teams working
 *  independently, but still have globally consistent rounds where row statuses change.
 */

template<typename device_t, typename rowmap_t, typename entries_t>
struct D2_MIS_Luby
{
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using bitset_t = Kokkos::Bitset<device_t>;
  using const_bitset_t = Kokkos::ConstBitset<device_t>;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  using lno_view_t = typename entries_t::non_const_type;
  //The type of status/priority values.
  using status_t = typename std::make_unsigned<lno_t>::type;
  using status_view_t = Kokkos::View<status_t*, mem_space>;
  using range_pol = Kokkos::RangePolicy<exec_space>;
  using team_pol = Kokkos::TeamPolicy<exec_space>;
  using team_mem = typename team_pol::member_type;

  KOKKOS_INLINE_FUNCTION static uint32_t xorshiftHash(uint32_t in)
  {
    uint32_t x = in;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
  }

  // Priority values 0 and max are special, they mean the vertex is
  // in the independent set or eliminated from consideration, respectively.
  // Values in between represent a priority for being added to the set,
  // based on degree and vertex ID as a tiebreak
  //   (higher priority = less preferred to being in the independent set)

  static constexpr status_t IN_SET = 0;
  static constexpr status_t OUT_SET = ~IN_SET;

  D2_MIS_Luby(const rowmap_t& rowmap_, const entries_t& entries_)
    : rowmap(rowmap_), entries(entries_), numVerts(rowmap.extent(0) - 1)
  {
    status_t i = numVerts + 1;
    nvBits = 0;
    while(i)
    {
      i >>= 1;
      nvBits++;
    }
    //Each value in rowStatus represents the status and priority of each row.
    //Each value in colStatus represents the lowest nonzero priority of any row adjacent to the column.
    //  This counts up monotonically as vertices are eliminated (given status OUT_SET)
    rowStatus = status_view_t(Kokkos::ViewAllocateWithoutInitializing("RowStatus"), numVerts);
    colStatus = status_view_t(Kokkos::ViewAllocateWithoutInitializing("ColStatus"), numVerts);
    allWorklists = Kokkos::View<lno_t**, Kokkos::LayoutLeft, mem_space>(Kokkos::ViewAllocateWithoutInitializing("AllWorklists"), numVerts, 3);
  }

  struct RefreshRowStatus
  {
    RefreshRowStatus(const status_view_t& rowStatus_, const lno_view_t& worklist_, lno_t nvBits_, int round)
      : rowStatus(rowStatus_), worklist(worklist_), nvBits(nvBits_)
    {
      hashedRound = xorshiftHash(round);
    }

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      //Combine vertex and round to get some pseudorandom priority bits that change each round
      status_t priority = xorshiftHash(i + hashedRound);
      //Generate unique status per row, with IN_SET < status < OUT_SET,
      int priorityBits = sizeof(status_t) * 8 - nvBits;
      status_t priorityMask = 1;
      priorityMask <<= priorityBits;
      priorityMask--;
      status_t newStatus = (status_t) (i + 1) + ((priority & priorityMask) << nvBits);
      if(newStatus == OUT_SET)
        newStatus--;
      rowStatus(i) = newStatus;
    }

    status_view_t rowStatus;
    lno_view_t worklist;
    int nvBits;
    uint32_t hashedRound;
  };

  struct RefreshColStatus
  {
    RefreshColStatus(const status_view_t& colStatus_, const lno_view_t& worklist_, const status_view_t& rowStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_)
      : colStatus(colStatus_), worklist(worklist_), rowStatus(rowStatus_), rowmap(rowmap_), entries(entries_), nv(nv_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      //iterate over {i} union the neighbors of i, to find
      //minimum status.
      status_t s = OUT_SET;
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      for(size_type j = rowBegin; j <= rowEnd; j++)
      {
        lno_t nei = (j == rowEnd) ? i : entries(j);
        if(nei <= nv)
        {
          status_t neiStat = rowStatus(nei);
          if(neiStat < s)
            s = neiStat;
        }
      }
      if(s == IN_SET)
        s = OUT_SET;
      colStatus(i) = s;
    }

    status_view_t colStatus;
    lno_view_t worklist;
    status_view_t rowStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
  };

  struct DecideSetFunctor
  {
    DecideSetFunctor(const status_view_t& rowStatus_, const status_view_t& colStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_, const lno_view_t& worklist_)
      : rowStatus(rowStatus_), colStatus(colStatus_), rowmap(rowmap_), entries(entries_), nv(nv_), worklist(worklist_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      //Processing row i.
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

    status_view_t rowStatus;
    status_view_t colStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
    lno_view_t worklist;
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
    InitWorklistFunctor(const lno_view_t& worklist_)
      : worklist(worklist_)
    {}
    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      worklist(i) = i;
    }
    lno_view_t worklist;
  };

  struct CompactWorklistFunctor
  {
    CompactWorklistFunctor(const lno_view_t& src_, const lno_view_t& dst_, const status_view_t& status_)
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

    lno_view_t src;
    lno_view_t dst;
    status_view_t status;
  };

  lno_view_t compute()
  {
    //Initialize first worklist to 0...numVerts
    lno_view_t rowWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 0);
    Kokkos::parallel_for(range_pol(0, numVerts), InitWorklistFunctor(rowWorklist));
    lno_view_t colWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 1);
    Kokkos::parallel_for(range_pol(0, numVerts), InitWorklistFunctor(colWorklist));
    lno_view_t thirdWorklist = Kokkos::subview(allWorklists, Kokkos::ALL(), 2);
    int round = 0;
    lno_t rowWorkLen = numVerts;
    lno_t colWorkLen = numVerts;
    while(true)
    {
      //Compute new row statuses
      Kokkos::parallel_for(range_pol(0, rowWorkLen), RefreshRowStatus(rowStatus, rowWorklist, nvBits, round));
      //Compute new col statuses
      Kokkos::parallel_for(range_pol(0, colWorkLen), RefreshColStatus(colStatus, colWorklist, rowStatus, rowmap, entries, numVerts));
      //Decide row statuses
      Kokkos::parallel_for(range_pol(0, rowWorkLen), DecideSetFunctor(rowStatus, colStatus, rowmap, entries, numVerts, rowWorklist));
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
  Kokkos::View<lno_t**, Kokkos::LayoutLeft, mem_space> allWorklists;
  //The number of bits required to represent vertex IDs, in the ECL-MIS tiebreak scheme:
  //  ceil(log_2(numVerts + 1))
  int nvBits;
  lno_t minDegree;
  lno_t maxDegree;
};

template<typename device_t, typename rowmap_t, typename entries_t>
struct D2_MIS_ECL
{
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using bitset_t = Kokkos::Bitset<device_t>;
  using const_bitset_t = Kokkos::ConstBitset<device_t>;
  using size_type = typename rowmap_t::non_const_value_type;
  using lno_t = typename entries_t::non_const_value_type;
  using lno_view_t = typename entries_t::non_const_type;
  //The type of status/priority values.
  using status_t = typename std::make_unsigned<lno_t>::type;
  using status_view_t = Kokkos::View<status_t*, mem_space>;
  using range_pol = Kokkos::RangePolicy<exec_space>;

  // Priority values 0 and max are special, they mean the vertex is
  // in the independent set or eliminated from consideration, respectively.
  // Values in between represent a priority for being added to the set,
  // based on degree and vertex ID as a tiebreak
  //   (higher priority = less preferred to being in the independent set)

  static constexpr status_t IN_SET = 0;
  static constexpr status_t OUT_SET = ~IN_SET;

  D2_MIS_ECL(const rowmap_t& rowmap_, const entries_t& entries_)
    : rowmap(rowmap_), entries(entries_), numVerts(rowmap.extent(0) - 1), colUpdateBitset(numVerts),
    worklist1(Kokkos::ViewAllocateWithoutInitializing("WL1"), numVerts),
    worklist2(Kokkos::ViewAllocateWithoutInitializing("WL2"), numVerts)
  {
    status_t i = numVerts + 1;
    nvBits = 0;
    while(i)
    {
      i >>= 1;
      nvBits++;
    }
    //Each value in rowStatus represents the status and priority of each row.
    //Each value in colStatus represents the lowest nonzero priority of any row adjacent to the column.
    //  This counts up monotonically as vertices are eliminated (given status OUT_SET)
    rowStatus = status_view_t(Kokkos::ViewAllocateWithoutInitializing("RowStatus"), numVerts);
    colStatus = status_view_t(Kokkos::ViewAllocateWithoutInitializing("ColStatus"), numVerts);
    KokkosKernels::Impl::graph_min_max_degree<device_t, lno_t, rowmap_t>(rowmap, minDegree, maxDegree);
    //Compute row statuses 
    Kokkos::parallel_for(range_pol(0, numVerts), InitRowStatus(rowStatus, rowmap, numVerts, nvBits, minDegree, maxDegree));
    //Compute col statuses
    Kokkos::parallel_for(range_pol(0, numVerts), InitColStatus(colStatus, rowStatus, rowmap, entries, numVerts));
  }

  struct InitRowStatus
  {
    InitRowStatus(const status_view_t& rowStatus_, const rowmap_t& rowmap_, lno_t nv_, lno_t nvBits_, lno_t minDeg_, lno_t maxDeg_)
      : rowStatus(rowStatus_), rowmap(rowmap_), nv(nv_), nvBits(nvBits_), minDeg(minDeg_), maxDeg(maxDeg_), invDegRange(1.f / (maxDeg - minDeg)) {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      //Generate unique status per row, with IN_SET < status < OUT_SET,
      int degBits = sizeof(status_t) * 8 - nvBits;
      if(degBits == 0)
      {
        //no space to store degree information. Algorithm will still work but will
        //probably produce a lower quality MIS.
        rowStatus(i) = i + 1;
        return;
      }
      status_t maxDegRange = (((status_t) 1) << degBits) - 2;
      lno_t deg = rowmap(i + 1) - rowmap(i);
      float degScore = (float) (deg - minDeg) * invDegRange;
      rowStatus(i) = (status_t) (i + 1) + (((status_t) (degScore * maxDegRange)) << nvBits);
    }

    status_view_t rowStatus;
    rowmap_t rowmap;
    lno_t nv;
    int nvBits;
    lno_t minDeg;
    lno_t maxDeg;
    float invDegRange;
  };

  struct InitColStatus
  {
    InitColStatus(const status_view_t& colStatus_, const status_view_t& rowStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_)
      : colStatus(colStatus_), rowStatus(rowStatus_), rowmap(rowmap_), entries(entries_), nv(nv_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      //iterate over {i} union the neighbors of i, to find
      //minimum status.
      status_t s = rowStatus(i);
      size_type rowBegin = rowmap(i);
      size_type rowEnd = rowmap(i + 1);
      for(size_type j = rowBegin; j < rowEnd; j++)
      {
        lno_t nei = entries(j);
        if(nei != i && nei < nv)
        {
          status_t neiStat = rowStatus(nei);
          if(neiStat < s)
            s = neiStat;
        }
      }
      colStatus(i) = s;
    }

    status_view_t colStatus;
    status_view_t rowStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
  };

  struct IterateStatusFunctor
  {
    IterateStatusFunctor(const status_view_t& rowStatus_, const status_view_t& colStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_, const lno_view_t& worklist_, const bitset_t& colUpdateBitset_)
      : rowStatus(rowStatus_), colStatus(colStatus_), rowmap(rowmap_), entries(entries_), nv(nv_), worklist(worklist_), colUpdateBitset(colUpdateBitset_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t i = worklist(w);
      //Processing row i.
      status_t s = rowStatus(i);
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
      bool statusChanged = neiOut || !neiMismatchS;
      if(neiOut)
      {
        //In order to make future progress, need to update the
        //col statuses for all neighbors of i which have status s.
        //This will increase the minimum to the next smallest row,
        //so that another nearby vertex can be added to the set.
        rowStatus(i) = OUT_SET;
      }
      else if(!neiMismatchS)
      {
        rowStatus(i) = IN_SET;
      }
      if(statusChanged)
      {
        for(size_type j = rowBegin; j <= rowEnd; j++)
        {
          lno_t nei = (j == rowEnd) ? i : entries(j);
          if(nei < nv && colStatus(nei) == s)
            colUpdateBitset.set(nei);
        }
      }
      //else: still undecided
    }

    status_view_t rowStatus;
    status_view_t colStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
    lno_view_t worklist;
    bitset_t colUpdateBitset;
  };

  struct UpdateWorklistFunctor
  {
    UpdateWorklistFunctor(const status_view_t& rowStatus_, const lno_view_t& oldWorklist_, const lno_view_t& newWorklist_)
      : rowStatus(rowStatus_), oldWorklist(oldWorklist_), newWorklist(newWorklist_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w, lno_t& lcount, bool finalPass) const
    {
      //processing row i
      lno_t i = oldWorklist(w);
      //Bit i will be set when it's decided IN_SET/OUT_SET.
      //If clear, vertex i needs to be processed still.
      status_t s = rowStatus(i);
      if(s != IN_SET && s != OUT_SET)
      {
        if(finalPass)
          newWorklist(lcount) = i;
        lcount++;
      }
    }

    status_view_t rowStatus;
    lno_view_t oldWorklist;
    lno_view_t newWorklist;
  };

  struct ColRefreshWorklist
  {
    ColRefreshWorklist(const bitset_t& colUpdateBitset_, const lno_view_t& refreshList_)
      : colUpdateBitset(colUpdateBitset_), refreshList(refreshList_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i, lno_t& lindex, bool finalPass) const
    {
      if(colUpdateBitset.test(i))
      {
        if(finalPass)
        {
          refreshList(lindex) = i;
          colUpdateBitset.reset(i);
        }
        lindex++;
      }
    }

    bitset_t colUpdateBitset;
    lno_view_t refreshList;
  };

  struct RefreshColStatus
  {
    RefreshColStatus(const lno_view_t& worklist_, const status_view_t& rowStatus_, const status_view_t& colStatus_, const rowmap_t& rowmap_, const entries_t& entries_, lno_t nv_)
      : worklist(worklist_), rowStatus(rowStatus_), colStatus(colStatus_), rowmap(rowmap_), entries(entries_), nv(nv_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t w) const
    {
      lno_t col = worklist(w);
      status_t minNeiStat = OUT_SET;
      size_type rowBegin = rowmap(col);
      size_type rowEnd = rowmap(col + 1);
      for(size_type j = rowBegin; j <= rowEnd; j++)
      {
        lno_t nei = (j == rowEnd) ? col : entries(j);
        if(nei >= nv)
          continue;
        status_t neiStat = rowStatus(nei);
        if(neiStat < minNeiStat)
          minNeiStat = neiStat;
      }
      if(minNeiStat == IN_SET)
        minNeiStat = OUT_SET;
      colStatus(col) = minNeiStat;
    }

    lno_view_t worklist;
    status_view_t rowStatus;
    status_view_t colStatus;
    rowmap_t rowmap;
    entries_t entries;
    lno_t nv;
  };

  struct InitWorklistFunctor
  {
    InitWorklistFunctor(const lno_view_t& worklist_)
      : worklist(worklist_)
    {}
    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      worklist(i) = i;
    }
    lno_view_t worklist;
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

  lno_view_t compute()
  {
    //Initialize first worklist to 0...numVerts
    Kokkos::parallel_for(range_pol(0, numVerts), InitWorklistFunctor(worklist1));
    lno_t workRemain = numVerts;
    int numIter = 0;
    while(workRemain)
    {
      //do another iteration
      Kokkos::parallel_for(range_pol(0, workRemain),
          IterateStatusFunctor(rowStatus, colStatus, rowmap, entries, numVerts, worklist1, colUpdateBitset));
      //And refresh the column statuses using the other worklist.
      lno_t colsToRefresh;
      Kokkos::parallel_scan(range_pol(0, numVerts),
          ColRefreshWorklist(colUpdateBitset, worklist2), colsToRefresh);
      Kokkos::parallel_for(range_pol(0, colsToRefresh),
          RefreshColStatus(worklist2, rowStatus, colStatus, rowmap, entries, numVerts));
      //then build the next worklist with a scan. Also get the length of the next worklist.
      lno_t newWorkRemain = 0;
      Kokkos::parallel_scan(range_pol(0, workRemain),
          UpdateWorklistFunctor(rowStatus, worklist1, worklist2),
          newWorkRemain);
      //Finally, flip the worklists
      std::swap(worklist1, worklist2);
      workRemain = newWorkRemain;
      numIter++;
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
  //The number of bits required to represent vertex IDs, in the ECL-MIS tiebreak scheme:
  //  ceil(log_2(numVerts + 1))
  int nvBits;
  lno_t minDegree;
  lno_t maxDegree;
  //Bitset representing columns whose status needs to be recomputed
  //These bits are cleared after each refresh.
  bitset_t colUpdateBitset;
  lno_view_t worklist1;
  lno_view_t worklist2;
};

}}}

#endif

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
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef _KOKKOS_SPGEMM_BMK_HPP
#define _KOKKOS_SPGEMM_BMK_HPP

#include "KokkosKernels_Handle.hpp"
#include "KokkosKernels_Sorting.hpp"
#include "KokkosSparse_marching_hash_table.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

// SpGEMM marching algorithm (symbolic, numeric1, triangle counting)
// - Allocate shared for hashtable
// - Until row is complete:
//  - Each thread (k lanes) consumes next k entries from each B row referenced by the A row.
//  - Entries are inserted into hash table in parallel, with linear probing or similar
//  - Threads record their minimum column with failing insertion, if any
//  - Team finds its minimum failing column c over all threads
//  - If any failure: flush all entries in the hashtable < c to the matrix, entry count or triangle count
//  - With new space cleared in hashtable, resume marching starting from c
//  - Problem: if a thread fails to insert most entries, progress will be slow.
//    Happens if a row of B starts earlier than others, but gets blocked out by rows which start later.
//    Choose a max column threshold and delete entries greater or equal, then retry.
//    Once all threads get past this threshold, raise it back to #cols.
//
//  Optimization ideas:
//   - Try to match team size to A avg nnz/row
//   - Each thread will be primary owner of 1 entry (row of B)
//    - Read row begin/end just once
//   - Whole team will process remaining entries in the slow path until complete
//   - In likely case, no extra global traffic at all compared to speck and old KK algorithm
//
//  Marching, numeric2
//   - Allocate shared for hashtable
//   - Until row complete:
//     - Insert contiguous chunk of C entries until hashtable at capacity (or insertion fails)
//     - Consume that column range from each B row, updating scalar values in the table
//     - Sort by keys and write values to C
//
//  Hash data structure that prioritizes minimum column indices?
//    - Each key could hash to some set of adresses (by linear probing, rehashing, etc)
//    - At each insertion step, have a bunch of hash addresses (some containing keys) and a bunch of
//      threads with (possibly new) keys to insert.
//    - For each key:
//      - If key already present, done
//      - If there is a free address, just insert normally
//      - If there are no free addresses, evict the greatest existing key in a matching slot and replace it
//      - Compute greatest existing key non-atomically and then compare-exchange?
//        - Don't want to use first-fit because it could evict smaller keys than needed, wasting hash capacity
//    - Benefits:
//      - At the end of process, hash table is likely to be nearly fully utilized (unless C row fits completely, which is also good)
//      - Guarantees forward progress (in fact, nearly the maximum possible progress for a given table size)
//      - For longer C rows (not fitting in hash table), global traffic is nearly minimized and coalesced (unlike a linked list or linear probe table in global)
//    - Downsides:
//      - How to resolve conflicts among threads inserting simultaneously?
//        - At least, this should be possible without touching global.
//      - The eviction requires some backtracking within each row (up to the chunk size)
//
//  Idea for optimization: many matrices have a high concentration of nonzeros very close to the diagonal.
//
//  Use a cycle:
//    - All threads attempt all insertions (with no evictions of existing keys). Only possible change for each table entry is EMPTY -> FULL.
//    - For threads where insertion failed, remember the maximum present table entry (eviction candidate)
//      - team barrier
//    - All threads whose entry to insert (X) is greater than that maximum must give up.
//    - For other threads, evict the maximum (and re-initialize its corresponding value). Now only possible change is FULL -> EMPTY.
//    - Tighten the maximum key under consideration to the minimum among all evictions and give-ups, minus 1.
//      - team barrier
//    - Once all entries in the A row have been processed in this way, table is left with some set of complete key-value pairs (up to the maximum under consideration).
//      - In symbolic, for all complete keys, just sum up the popcounts of the corresponding values.
//      - In numeric1, bitonic sort the whole table and just blit out the complete columns/values to C
//    - Use ArithTraits::max as the special key representing EMPTY. That way, after sorting those are all at the end.
//
//  IMPORTANT FOR PERFORMANCE:
//    If #threads >= nnz(aRow), then each thread should just own a row of B and hold the marching iterator and row begin/end in register.
//    Otherwise, would be forced to read+write a whole extra cacheline from global each step - marchIterators(aEntry)
//
//  Tuning inputs:
//    - A avg nz/row
//    - B avg nz/row
//    - A max nz/row? B max nz/row? Both computable quickly with an extra reduction
//  Tuning outputs:
//    - Hash table size (balance capacity and occupancy)
//    - Linear probing attempts
//    - Team size (ideally at least A avg nz/row)
//    - Vector length (should be roughly B nz/row)
//
//MARCHING PSEUDOCODE
//
//For each row of A:
//  While not all entries of A/rows of B have been marched through:
//    Take batches of B entries starting at march position for the A entry
//    Attempt to insert the entries. Keep track of failures (due to the table being full) and evictions
//  For each thread, find the minimum insertion failure (if none, ORDINAL_MAX)
//    Then find the min of this over all threads
//  Find the min failure over all threads, and min evicted key (both are ORDINAL_MAX if none)
//  Go through the table. Copy out keys < min failure to the output, and clear keys >= min eviction (non-secured)
//  Now only partially computed (but secured) keys remain.
//  Advance march positions by the threadsize, wherever the final column in batch is secured.

namespace KokkosSparse {
namespace Impl {
  template<typename Ordinal>
  struct SpgemmTeamInfo
  {
    KOKKOS_INLINE_FUNCTION SpgemmTeamInfo() = default;

    KOKKOS_INLINE_FUNCTION SpgemmTeamInfo(Ordinal workRemains_, Ordinal minFail_, Ordinal minEviction_)
      : workRemains(workRemains_), minFail(minFail_), minEviction(minEviction_)
    {}

    //use operator+ to simultanesouly sum-reduce workRemains, and min-reduce minFail and minEviction
    KOKKOS_INLINE_FUNCTION friend SpgemmTeamInfo<Ordinal> operator+(SpgemmTeamInfo<Ordinal> lhs, const SpgemmTeamInfo<Ordinal>& rhs)
    {
      lhs.workRemains += rhs.workRemains;
      if(rhs.minFail < lhs.minFail)
        lhs.minFail = rhs.minFail;
      if(rhs.minEviction < lhs.minEviction)
        lhs.minEviction = rhs.minEviction;
      return lhs;
    }

    KOKKOS_INLINE_FUNCTION SpgemmTeamInfo<Ordinal>& operator+=(const SpgemmTeamInfo<Ordinal>& other)
    {
      workRemains += other.workRemains;
      if(other.minFail < minFail)
        minFail = other.minFail;
      if(other.minEviction < minEviction)
        minEviction = other.minEviction;
      return *this;
    }

    Ordinal workRemains;
    Ordinal minFail;
    Ordinal minEviction;
  };
}
}

namespace Kokkos {
  template<typename Ordinal>
  struct reduction_identity<KokkosSparse::Impl::SpgemmTeamInfo<Ordinal>> {
    KOKKOS_FORCEINLINE_FUNCTION constexpr static KokkosSparse::Impl::SpgemmTeamInfo<Ordinal> sum()
    {
      Ordinal mx = Kokkos::ArithTraits<Ordinal>::max();
      return KokkosSparse::Impl::SpgemmTeamInfo<Ordinal>(0, mx, mx);
    }
  };
}

namespace KokkosSparse {
namespace Impl {

template<typename Policy, typename RowmapIn, typename RowmapOut, typename Entries, typename OrdinalView>
struct SpGEMMSymbolicFunctor
{
  using TeamMem = typename Policy::member_type;
  using Offset = typename RowmapOut::non_const_value_type;
  using Ordinal = typename Entries::non_const_value_type;
  using AT = Kokkos::ArithTraits<Ordinal>;

  SpGEMMSymbolicFunctor(const RowmapIn& aRowmap_, const Entries& aEntries_, const RowmapIn& bRowmap_, const Entries& bEntries_, const RowmapOut& cRowmap_, const OrdinalView& marchIterators_, const OrdinalView& batchEnds_, int hashSize_, int vectorLen_)
    : aRowmap(aRowmap_), aEntries(aEntries_), bRowmap(bRowmap_), bEntries(bEntries_), cRowmap(cRowmap_), marchIterators(marchIterators_), batchEnds(batchEnds_), hashSize(hashSize_), vectorLen(vectorLen_)
  {}

  KOKKOS_INLINE_FUNCTION void operator()(const TeamMem& t) const
  {
    //if(t.league_rank() != 17)
    //  return;
    Offset aRowBegin = aRowmap(t.league_rank());
    Ordinal aRowLen = aRowmap(t.league_rank() + 1) - aRowBegin;
    //Counting number of entries in C row
    Ordinal numEntries = 0;
    //Acquire memory for the hash table
    //In symbolic, hashtable keys are columns divided by 32.
    //Values are 32-wide bitsets - 1 represents a present entry, 0 represents no entry.
    //TODO: autotune nprobe? Runtime or compile-time?
    std::cout << "Hash table has " << hashSize << " slots.\n";
    MarchingHashTable<Ordinal, uint32_t> ht(
        (Ordinal*) t.team_shmem().get_shmem(hashSize * sizeof(Ordinal)),
        (uint32_t*) t.team_shmem().get_shmem(hashSize * sizeof(uint32_t)),
        hashSize, 4);
    //Mark all hash keys as empty (represented using Ordinal's max value)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, hashSize),
      [&](int i)
      {
        ht.keys[i] = AT::max();
      });
    t.team_barrier();
    int numThreads = t.team_size() / vectorLen;
    int tid = t.team_rank() / vectorLen;
    int vid = t.team_rank() % vectorLen;
    //Team-wide column window that is currently being processed.
    //Cols < completedCol have already been inserted in table, counted and then cleared.
    //Cols between completedCol and securedCol are partially computed in the table (none evicted)
    Ordinal completedCol = 0;
    Ordinal securedCol = 0;
    bool firstRound = true;
    //Loop until all entries of each referenced row of B have been consumed
    int round = 0;
    while(true)
    {
      std::cout << "\n\nAt the beginning of round " << round << ", completedCol = " << completedCol << " and securedCol = " << securedCol << '\n';
      Ordinal localMinFail = AT::max();
      Ordinal localMinEviction = AT::max();
      int threadWorkRemains = 0;
      //In this loop, threadWorkRemains can only be increased to 1, never reset to 0
      for(Ordinal aIter = 0; aIter < aRowLen; aIter += numThreads)
      {
        Ordinal bRow, bCol;
        Offset bRowBegin;
        Ordinal bRowLen;
        //This absolute index into A's entries, as well as the marching iterators
        //is used several times.
        Offset aEntryIndex = aRowBegin + aIter + tid;
        bool threadActive = aIter + tid < aRowLen;
        Ordinal marchPos;
        if(threadActive)
        {
          bRow = aEntries(aEntryIndex);
          bRowBegin = bRowmap(bRow);
          bRowLen = bRowmap(bRow + 1) - bRowBegin;
          marchPos = marchIterators(aEntryIndex);
          if(marchPos == bRowLen)
          {
            //Previously finished marching through this row of B
            std::cout << ">> Thread " << t.team_rank() << " inactive because march pos for B row " << bRow << " is the same as the B row's length (" << bRowLen << ")\n";
            threadActive = false;
          }
          else
          {
            //Still (might be) work to do
            Ordinal batchEnd = batchEnds(aEntryIndex);
            //If c <= securedCol, then any thread which previously attempted to insert c must have succeeded.
            if(!firstRound && batchEnd <= securedCol)
            {
              //Go to the next batch, and update both marchIterators and batchEnds
              //(batchEnds must be updated after actually loading the new batch of columns)
              marchPos += vectorLen;
              if(marchPos > bRowLen)
                marchPos = bRowLen;
              marchIterators(aEntryIndex) = marchPos;
              threadActive = marchPos + vid < bRowLen;
              if(!threadActive)
                std::cout << ">> Thread " << t.team_rank() << " inactive because after advacing iter, batch extends past end of B row (march pos = " << marchPos << ", bRowLen = " << bRowLen << ")\n";
              if(threadActive)
              {
                bCol = bEntries(bRowBegin + marchPos + vid);
                threadWorkRemains = 1;
              }
              //If I am the last active vector lane now, bCol is the new end of batch
              if((marchPos + vectorLen < bRowLen && vid == vectorLen - 1) ||
                  marchPos + vid == bRowLen - 1)
              {
                batchEnds(aEntryIndex) = bCol;
              }
            }
            else
            {
              //Working with same batch again
              threadActive = marchPos + vid < bRowLen;
              if(!threadActive)
                std::cout << ">> Thread " << t.team_rank() << " inactive because batch extends past end of B row (march pos = " << marchPos << ", bRowLen = " << bRowLen << ", vid = " << vid << ")\n";
              if(threadActive)
              {
                bCol = bEntries(bRowBegin + marchPos + vid);
                threadWorkRemains = 1;
              }
            }
            //If I am the last active vector lane AND the batch doesn't reach the end of B's row,
            //make sure localMinFail reflects the fact that no columns beyond this batch can't be complete
            if(vid == vectorLen - 1 && threadActive)
            {
              if(bCol + 1 < localMinFail)
                localMinFail = bCol + 1;
            }
          }
        }
        if(threadActive)
        {
          //Thread is still active, so it has an entry of B to read and attempt to insert.
          //completedCol is initially 0, so there is a special case for bCol == 0.
          threadActive = bCol > completedCol || (bCol == 0 && completedCol == 0);
          if(!threadActive)
            std::cout << ">> Thread " << t.team_rank() << " inactive because column " << bCol << " is less than or equal to completedCol " << completedCol << '\n';
        }
        if(threadActive)
        {
          //std::cout << "  ** HELLO: I am thread " << t.team_rank() << " and I am active with bCol = " << bCol << '\n';
        }
        else
        {
          std::cout << "  ** HELLO: I am thread " << t.team_rank() << " and I am inactive.\n";
        }
        if(threadActive)
        {
          Ordinal eviction = ht.insert(bCol / 32, securedCol / 32 + 1);
          if(eviction < localMinEviction)
            localMinEviction = eviction;
        }
        //Team-wide barrier, to allow all insertion attempts to finish
        //(this must involve every single thread, which is why threadActive is necessary)
        t.team_barrier();
        if(threadActive)
        {
          //std::cout << "Testing whether col " << bCol << " (key " << bCol / 32 << ") made it in, and if so setting bit.\n";
          Ordinal key = bCol / 32;
          if(!ht.updateValueOr(key, 1U << (bCol % 32)))
          {
            //The key did not make it into the table
            if(bCol < localMinFail)
              localMinFail = bCol;
            std::cout << "  Row " << t.league_rank() << ": FAILED to insert column " << bCol << " (key " << bCol / 32 << ")\n";
          }
          else
          {
            std::cout << "  Row " << t.league_rank() << ": successfully inserted column " << bCol << " (key " << bCol / 32 << ")\n";
          }
        }
      }
      std::cout << "March iterators after this round: ";
      for(Offset asdf = aRowBegin; asdf < aRowBegin + aRowLen; asdf++)
        std::cout << marchIterators(asdf) << ' ';
      std::cout << '\n';
      //Need to do 3 reductions now:
      //  - Figure out if the row is done (true if threadWorkRemains is 0 on all threads)
      //  - Update completedCol based on localMinFail
      //  - Update securedCol based on localMinEviction
      SpgemmTeamInfo<Ordinal> teamInfo;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, t.team_size()),
        [&](int, SpgemmTeamInfo<Ordinal>& linfo)
        {
          if(threadWorkRemains)
            linfo.workRemains = 1;
          if(localMinFail < linfo.minFail)
            linfo.minFail = localMinFail;
          if(localMinEviction < linfo.minEviction)
            linfo.minEviction = localMinEviction;
        }, teamInfo);
      completedCol = teamInfo.minFail - 1;
      securedCol = teamInfo.minEviction - 1;
      Kokkos::single(Kokkos::PerTeam(t),
        [=]()
        {
          std::cout << "Row " << t.league_rank() << ": minimum excluded key: " << teamInfo.minFail << " and min evicted key: " << teamInfo.minEviction << '\n';
        std::cout << "Row " << t.league_rank() << ": next batch will insert keys starting with " << securedCol + 1 << '\n';
        });
      //Traverse hash table and count the entries, up to the beginCol for the next iter
      Ordinal iterNumEntries;
      //What is the maximum key which could contain completed columns?
      Ordinal maxCompletedKey = completedCol / 32;
      std::cout << "NOTE: maxCompletedKey = " << maxCompletedKey << ", corresponds to columns " << maxCompletedKey * 32 << "..." << maxCompletedKey*32 + 31 << " inclusive.\n";
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, hashSize),
        [&](int i, Ordinal& lcount)
        {
          Ordinal key = ht.keys[i];
          if(key <= maxCompletedKey)
          {
            uint32_t mask;
            if(key * 32 > maxCompletedKey)
              mask = ~uint32_t(0);
            else
              mask = (uint32_t(1) << (completedCol + 1 - key * 32)) - 1;
            uint32_t bitsToCount;
            if(ht.values[i] & ~mask)
            {
              //This key has some non-complete columns, so don't remove the key.
              //Only zero out the bits corresponding to complete columns in the value.
              bitsToCount = ht.values[i] & mask;
              ht.values[i] &= ~mask;
            }
            else
            {
              bitsToCount = ht.values[i];
              ht.keys[i] = AT::max();
            }
            for(int asdf = 0; asdf < 32; asdf++)
            {
              if(bitsToCount & (uint32_t(1) << asdf))
                std::cout << "Peeling off completed column " << key * 32 + asdf << '\n';
            }
            lcount += KokkosKernels::Impl::pop_count(bitsToCount);
          }
        }, iterNumEntries);
      numEntries += iterNumEntries;
      if(!teamInfo.workRemains)
      {
        //Processing of all rows of B is done
        break;
      }
      round++;
      firstRound = false;
    }
    Kokkos::single(Kokkos::PerTeam(t),
      [&]()
      {
        cRowmap(t.league_rank()) = numEntries;
      });
  }

  RowmapIn aRowmap;
  Entries aEntries;
  RowmapIn bRowmap;
  Entries bEntries;
  RowmapOut cRowmap;
  OrdinalView marchIterators;
  OrdinalView batchEnds;
  int hashSize;
  int vectorLen;
};

//A is m x n
//B is n x k
//C is m x k
template<typename KernelHandle, typename RowmapIn, typename RowmapOut, typename Entries>
void bmk_SpGEMM_Symbolic(int m, int n, int k, KernelHandle* handle, const RowmapIn& aRowmap, const Entries& aEntries, const RowmapIn& bRowmap, const Entries& bEntries, const RowmapOut& cRowmap)
{
  using ExecSpace = typename KernelHandle::HandleExecSpace;
  using Policy = Kokkos::TeamPolicy<ExecSpace>;
  using Offset = typename RowmapOut::non_const_value_type;
  using Ordinal = typename Entries::non_const_value_type;
  using OrdinalView = Kokkos::View<Ordinal*, typename KernelHandle::HandleTempMemorySpace>;
  //Allocate the marching counters array
  OrdinalView marchIterators("Marching Iterators", aEntries.extent(0));
  OrdinalView batchEnds("Batch ends", aEntries.extent(0));
  //Choose tunable parameters: team size, vector length and hash table size.
  //(team size) * (vector length) is constrained by max block size.
  //Hash table size is constrained by shared memory.
  //Team size should ideally be >= avg A nnz/row.
  //Vector length should ideally be >= avg B nnz/row.
  //Hash table size is harder to estimate as it depends on term compaction. Too big = low occupancy, too small = slower marching progress.
  //  Also, there is some work done in shared memory that is proportional to total table size, not just number of filled cells.

  //int teamSize = 16;
  //int vectorLength = 16;
  int teamSize = 1;
  int vectorLength = 1;
  int hashSize = 512;
  SpGEMMSymbolicFunctor<Policy, RowmapIn, RowmapOut, Entries, OrdinalView> functor(aRowmap, aEntries, bRowmap, bEntries, cRowmap, marchIterators, batchEnds, hashSize, vectorLength);
  Policy pol(m, teamSize * vectorLength);
  pol.set_scratch_size(0, Kokkos::PerTeam(hashSize * (sizeof(Ordinal) + sizeof(uint32_t))));
  Kokkos::parallel_for(pol, functor);
  //Then exclusive prefix-sum cRowmap, and give the handle the total number of C entries.
  Offset c_nnz;
  KokkosKernels::Impl::kk_exclusive_parallel_prefix_sum<RowmapOut, ExecSpace>(m + 1, cRowmap, c_nnz);
  handle->get_spgemm_handle()->set_c_nnz(c_nnz);
}

} // Impl
} // KokkosSparse

#endif


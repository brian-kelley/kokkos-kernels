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
#include "Kokkos_ArithTraits.hpp"
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
//    If #threads >= nnz(aRow), then each thread should own a row of B and hold the marching iterator and row begin/end in register.
//    Otherwise, would be forced to read an extra line from global each step.
//
//  Tuning inputs: A avg nz/row, B avg nz/row
//  Tunable parameters:
//    - Hash table size (balance capacity and occupancy)
//    - Linear probing attempts
//    - Team size (ideally at least A avg nz/row)
//    - Vector length (should be roughly B nz/row)

namespace KokkosSparse {
namespace Impl {

template<typename Policy, typename Rowmap, typename Entries, typename MarchIterators>
struct SpGEMMSymbolicFunctor
{
  using TeamMem = typename Policy::member_type;
  using SizeType = typename Rowmap::non_const_value_type;
  using Ordinal = typename Entries::non_const_value_type;
  using AT = Kokkos::ArithTraits<Ordinal>;

  SpGEMMSymbolicFunctor(const Rowmap& aRowmap_, const Entries& aEntries_, const Rowmap& bRowmap_, const Entries& bEntries_, const Rowmap& cRowmap_, const MarchIterators& marchIterators_, Ordinal bCols_, int hashSize_, int vectorLen_)
    : aRowmap(aRowmap_), aEntries(aEntries_), bRowmap(bRowmap_), bEntries(bEntries_), cRowmap(cRowmap_), marchIterators(marchIterators_), bCols(bCols_), hashSize(hashSize_), vectorLen(vectorLen_)
  {}

  KOKKOS_INLINE_FUNCTION void operator(const TeamMem& t) const
  {
    SizeType aRowBegin = aRowmap(t.league_rank());
    Ordinal aRowLen = aRowmap(t.league_rank() + 1) - aRowBegin;
    //Counting number of entries in C row
    //Only needs to be meaningful inside team-wide single (it's computed via reductions)
    Ordinal numEntries = 0;
    //Acquire memory for the hash table
    //In symbolic, hashtable keys are columns divided by 32.
    //Values are 32-wide bitsets - 1 represents a present entry, 0 represents no entry.
    //TODO: autotune nprobe? Should stay compile-time constant?
    MarchingHashTable<Ordinal, uint32_t> ht(
        (Ordinal*) t.team_shmem().get_shmem(hashSize * sizeof(Ordinal)),
        (uint32_t*) t.team_shmem().get_shmem(hashSize * sizeof(uint32_t)),
        hashSize, 4);
    //Mark all hash keys as empty (this constant is just Ordinal's max value)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, hashSize),
      [&](int i)
      {
        ht.keys[i] = AT::max();
      });
    t.team_barrier();
    int numThreads = t.team_size() / vectorLength;
    int tid = t.team_rank() / vectorLength;
    int vid = t.team_rank() % vectorLength;
    //Column window that is currently being processed.
    //Cols < beginCol have already been inserted in table, counted and then cleared.
    Ordinal beginCol = 0;
    Ordinal teamMinFail;
    //Loop until all entries of each referenced row of B have been consumed
    while(true)
    {
      teamMinFail = AT::max();
      for(Ordinal aIter = 0; aIter < aRowLen; aIter += numThreads)
      {
        Ordinal bRow, bCol;
        SizeType bRowBegin;
        Ordinal bRowLen;
        //This absolute index into A's entries, as well as the marching iterators
        //is used several times.
        SizeType aEntryIndex = aRowBegin + aIter + tid;
        Ordinal marchPos;
        bool threadActive = aIter + tid < aRowLen;
        if(threadActive)
        {
          bRow = aEntries(aEntryIndex);
          bRowBegin = bRowmap(bRow);
          bRowLen = bRowmap(bRow + 1) - bRowBegin;
          //marchPos contains pairs, one corresponding to each entry in A (a reference to row of B).
          // First: the index from bRowBegin where entries will be consumed.
          //   This is incremented before consuming entries, but only if
          //   the marching of the last iteration got past the last column in this batch.
          // Second: The last column in the current batch. This is updated after reading in all the entries of B.
          //
          // Decide whether marching index can advance.
          marchPos = marchIterators(aEntryIndex).first;
          if(marchPos < bRowLen && beginCol > marchIterators(aEntryIndex).second)
          {
            //Can advance the marching position for this entry
            marchPos += vectorLength;
            if(marchPos > bRowLen)
              marchPos = bRowLen;
            if(vid == 0)
              marchIterators(aEntryIndex).first = marchPos;
          }
          //Now (for this vector lane), check if there is an entry of B to consume
          threadActive = marchPos + vid < bRowLen;
        }
        if(threadActive)
        {
          //Thread is still active, so it has an entry of B to read and attempt to insert.
          bCol = bEntries(bRowBegin + marchPos + vid);
          threadActive = bCol >= beginCol;
          //Also update the second marching value (last valid column in batch)
          Ordinal lastActiveVectorLane = vectorLength - 1;
          if(marchPos + vectorLength > bRowLen)
          {
            lastActiveVectorLane = bRowLen - marchPos;
          }
          if(vid == lastActiveVectorLane)
          {
            marchIterators(aEntryIndex).second = bCol;
          }
        }
        if(threadActive)
          ht.insert(bCol / 32);
        //Team-wide barrier, to allow all insertion attempts to finish
        //(this must involve every single thread, which is why threadActive is necessary)
        t.team_barrier();
        Ordinal failingKey = AT::max();
        if(threadActive)
        {
          if(!ht.updateValueOr(bCol / 32, 1U << (bCol % 32)))
          {
            //The key did not make it into the table
            failingKey = bCol / 32;
          }
        }
        Ordinal batchMinFail;
        //Find the minimum failing key over all threads, and update the global minimum for this march iteration
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, t.team_size()),
        [&](int, Ordinal& lminFail)
        {
          If(failingKey < lminFail)
            lminFail = failingKey;
        }, Kokkos::Min<Ordinal>(batchMinFail));
        //Since batchMinFail is the result of team reduction, it's only
        //present on thread 0. So is teamMinFail.
        Kokkos::single(Kokkos::PerTeam(t),
        [&]()
        {
          if(batchMinFail < teamMinFail)
            teamMinFail = batchMinFail;
        });
      }
      //Finally, after visiting every referenced row of B, update beginCol for the next march iteration.
      //Must be computed on thread 0, and then broadcast to the rest of the team.
      Kokkos::single(Kokkos::PerTeam(t),
        [&](Ordinal& lbeginCol)
        {
          lbeginCol = teamMinFail * 32;
        }, beginCol);
      //Traverse hash table and count the entries, up to the beginCol for the next iter
      Ordinal iterNumEntries;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, hashSize),
        [&](int i, Ordinal& lcount)
        {
          if(ht.keys[i] * 32 < beginCol)
            lcount += KokkosKernels::Impl::pop_count(ht.values[i]);
          //In any case, re-initialize the key to empty
          ht.keys[i] = AT::max();
        }, iterNumEntries);
      //Note: numEntries is still only significant on thread 0
      numEntries += iterNumEntries;
    }
    Kokkos::single(Kokkos::PerTeam(t),
      [&]()
      {
        cRowmap(t.league_rank()) = numEntries;
      });
  }

  Rowmap aRowmap;
  Entries aEntries;
  Rowmap bRowmap;
  Entries bEntries;
  Rowmap cRowmap;
  MarchIterators marchIterators;
  Ordinal bCols;
  int hashSize;
  int vectorLen;
};

//A is m x n
//B is n x k
//C is m x k
template<typename KernelHandle, typename Rowmap, typename Entries>
void bmk_SpGEMM_Symbolic(int m, int n, int k, KernelHandle* handle, const Rowmap& aRowmap, const Entries& aEntries, const Rowmap& bRowmap, const Entries& bEntries, const Rowmap& cRowmap)
{
  using ExecSpace = typename KernelHandle::HandleExecSpace;
  using Policy = Kokkos::TeamPolicy<ExecSpace>;
  using MarchIterators = Kokkos::View<Kokkos::pair<Ordinal, Ordinal>*, typename KernelHandle::HandleTempMemorySpace>;
  using size_type = typename Rowmap::non_const_value_type;
  //Allocate the marching counters array
  MarchIterators marchIterators("Marching Iterators", aEntries.extent(0));
  //Choose tunable parameters: team size, vector length and hash table size.
  //(team size) * (vector length) is constrained by max block size.
  //Hash table size is constrained by shared memory.
  //Team size should ideally be >= avg A nnz/row.
  //Vector length should ideally be >= avg B nnz/row.
  //Hash table size is harder to estimate as it depends on term compaction. Too big = low occupancy, too small = slower marching progress.
  //  Also, there is some work done in shared memory that is proportional to total table size, not just number of filled cells.
  int teamSize = 16;
  int vectorLength = 16;
  int hashSize = 512;
  SpGEMMSymbolicFunctor<Policy, Rowmap, Entries, MarchIterators> functor(aRowmap, aEntries, bRowmap, bEntries, cRowmap, marchIterators, k, hashSize, vectorLength);
  Policy pol(m, teamSize * vectorLength);
  pol.set_scratch_size(0, Kokkos::PerTeam(hashSize * (sizeof(Ordinal) + sizeof(uint32_t))));
  Kokkos::parallel_for(pol, functor);
  //Then exclusive prefix-sum cRowmap, and give the handle the total number of C entries.
  size_type c_nnz;
  KokkosKernels::Impl::kk_exclusive_parallel_prefix_sum<Rowmap, ExecSpace>(m + 1, cRowmap, c_nnz);
  handle->get_spgemm_handle()->set_c_nnz(c_nnz);
}

} // Impl
} // KokkosSparse

#endif


/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#include "KokkosKernels_Utils.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_Parallel_Reduce.hpp>
#include "KokkosGraph_graph_color.hpp"
#include "KokkosKernels_Uniform_Initialized_MemoryPool.hpp"
#ifndef _KOKKOSRCMIMP_HPP
#define _KOKKOSRCMIMP_HPP

namespace KokkosSparse{

namespace Impl{

template <typename HandleType, typename lno_row_view_t, typename lno_nnz_view_t>
struct RCM
{
  enum PeripheralMode
  {
    MIN_DEGREE,
    DOUBLE_BFS
  };

  typedef typename HandleType::HandleExecSpace MyExecSpace;
  typedef typename HandleType::HandleTempMemorySpace MyTempMemorySpace;
  typedef typename HandleType::HandlePersistentMemorySpace MyPersistentMemorySpace;

  typedef typename HandleType::size_type size_type;
  typedef typename HandleType::nnz_lno_t nnz_lno_t;

  typedef typename lno_row_view_t::const_type const_lno_row_view_t;
  typedef typename lno_row_view_t::non_const_type non_const_lno_row_view_t;
  typedef typename non_const_lno_row_view_t::value_type offset_t;

  typedef typename lno_nnz_view_t::non_const_type non_const_lno_nnz_view_t;

  typedef typename HandleType::row_lno_temp_work_view_t row_lno_temp_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_view_t row_lno_persistent_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_host_view_t row_lno_persistent_work_host_view_t; //Host view type

  typedef typename HandleType::nnz_lno_temp_work_view_t nnz_lno_temp_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_host_view_t nnz_lno_persistent_work_host_view_t; //Host view type

  typedef Kokkos::View<nnz_lno_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0>> perm_view_t;
  typedef Kokkos::View<nnz_lno_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0>> result_view_t;
  typedef Kokkos::View<nnz_lno_t, MyTempMemorySpace, Kokkos::MemoryTraits<0>> single_view_t;

  typedef Kokkos::RangePolicy<MyExecSpace> my_exec_space;

  typedef Kokkos::RangePolicy<MyExecSpace> range_policy_t ;
  typedef Kokkos::TeamPolicy<MyExecSpace> team_policy_t ;
  typedef typename team_policy_t::member_type team_member_t ;

  typedef Kokkos::MinLoc<nnz_lno_t, nnz_lno_t, MyTempMemorySpace> MinLocReducer;
  typedef Kokkos::MaxLoc<nnz_lno_t, nnz_lno_t, MyTempMemorySpace> MaxLocReducer;
  typedef Kokkos::ValLocScalar<nnz_lno_t, nnz_lno_t> ValLoc;

  typedef nnz_lno_t LO;

  RCM(HandleType* handle_, size_type numRows_, lno_row_view_t rowmap_, lno_nnz_view_t colinds_)
    : handle(handle_), numRows(numRows_),
      rowmap(rowmap_), colinds(colinds_)
  {}

  HandleType* handle;
  size_type numRows;
  lno_row_view_t rowmap;
  lno_nnz_view_t colinds;

  //simple parallel reduction to find max degree in graph
  nnz_lno_t find_max_degree()
  {
    offset_t maxDeg = 0;
    Kokkos::parallel_reduce(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(nnz_lno_t i, offset_t& lmaxDeg)
      {
        offset_t nnz = rowmap(i + 1) - rowmap(i);
        if(nnz > lmaxDeg)
        {
          lmaxDeg = nnz;
        }
      }, Kokkos::Max<offset_t>(maxDeg));
    //max degree should be computed as an offset_t,
    //but must fit in a nnz_lno_t
    return (nnz_lno_t) maxDeg;
  }

  nnz_lno_t find_bandwidth(lno_row_view_t rowptrs, lno_nnz_view_t colinds)
  {
    nnz_lno_t maxBand = 0;
    Kokkos::parallel_reduce(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(nnz_lno_t i, nnz_lno_t& lmaxBand)
      {
        for(size_type j = rowptrs(i); j < rowptrs(i + 1); j++)
        {
          nnz_lno_t thisBand = colinds(j) - i;
          if(thisBand < 0)
            thisBand = -thisBand;
          if(thisBand > lmaxBand)
          {
            lmaxBand = thisBand;
          }
        }
      }, Kokkos::Max<nnz_lno_t>(maxBand));
    return maxBand;
  }

  double find_average_bandwidth(lno_row_view_t rowptrs, lno_nnz_view_t colinds)
  {
    size_type totalBand = 0;
    Kokkos::parallel_reduce(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(nnz_lno_t i, size_type& lsum)
      {
        for(offset_t j = rowptrs(i); j < rowptrs(i + 1); j++)
        {
          if(colinds(j) < i)
            lsum += i - colinds(j);
          else
            lsum += colinds(j) - i;
        }
      }, Kokkos::Sum<size_type>(totalBand));
    return (double) totalBand / rowptrs(numRows);
  }

  template<typename size_type, typename KeyType, typename ValueType, typename IndexType>
  KOKKOS_INLINE_FUNCTION void
  radixSortKeysAndValues(KeyType* keys, KeyType* keysAux, ValueType* values, ValueType* valuesAux, IndexType n)
  {
    if(n <= 1)
      return;
    //sort 4 bits at a time
    KeyType mask = 0xF;
    bool inAux = false;
    //maskPos counts the low bit index of mask (0, 4, 8, ...)
    IndexType maskPos = 0;
    IndexType sortBits = 0;
    KeyType minKey = keys[0];
    KeyType maxKey = keys[0];
    for(size_type i = 0; i < n; i++)
    {
      if(keys[i] < minKey)
        minKey = keys[i];
      if(keys[i] > maxKey)
        maxKey = keys[i];
    }
    //subtract a bias of minKey so that key range starts at 0
    for(size_type i = 0; i < n; i++)
    {
      keys[i] -= minKey;
    }
    KeyType upperBound = maxKey - minKey;
    while(upperBound)
    {
      upperBound >>= 1;
      sortBits++;
    }
    for(size_type s = 0; s < (sortBits + 3) / 4; s++)
    {
      //Count the number of elements in each bucket
      IndexType count[16] = {0};
      IndexType offset[17] = {0};
      if(!inAux)
      {
        for(IndexType i = 0; i < n; i++)
        {
          count[(keys[i] & mask) >> maskPos]++;
        }
      }
      else
      {
        for(IndexType i = 0; i < n; i++)
        {
          count[(keysAux[i] & mask) >> maskPos]++;
        }
      }
      //get offset as the prefix sum for count
      for(IndexType i = 0; i < 16; i++)
      {
        offset[i + 1] = offset[i] + count[i];
      }
      //now for each element in [lo, hi), move it to its offset in the other buffer
      //this branch should be ok because whichBuf is the same on all threads
      if(!inAux)
      {
        //copy from *Over to *Aux
        for(IndexType i = 0; i < n; i++)
        {
          IndexType bucket = (keys[i] & mask) >> maskPos;
          keysAux[offset[bucket + 1] - count[bucket]] = keys[i];
          valuesAux[offset[bucket + 1] - count[bucket]] = values[i];
          count[bucket]--;
        }
      }
      else
      {
        //copy from *Aux to *Over
        for(IndexType i = 0; i < n; i++)
        {
          IndexType bucket = (keysAux[i] & mask) >> maskPos;
          keys[offset[bucket + 1] - count[bucket]] = keysAux[i];
          values[offset[bucket + 1] - count[bucket]] = valuesAux[i];
          count[bucket]--;
        }
      }
      inAux = !inAux;
      mask = mask << 4;
      maskPos += 4;
    }
    if(inAux)
    {
      //need to deep copy from aux arrays to main
      for(IndexType i = 0; i < n; i++)
      {
        keys[i] = keysAux[i];
        values[i] = valuesAux[i];
      }
    }
    //remove the bias
    for(size_type i = 0; i < n; i++)
    {
      keys[i] += minKey;
    }
  }

  //breadth-first search, producing a Cuthill-McKee ordering
  //nodes are labeled in the exact order they would be in a serial BFS,
  //and neighbors are visited in ascending order of degree
  result_view_t parallel_rcm_bfs(nnz_lno_t start)
  {
  /*
    std::cout << "In rcm().\n";
    std::cout << "# Entries per row (check for weird values):\n";
    for(nnz_lno_t i = 0; i < numRows; i++)
    {
      std::cout << rowmap(i + 1) - rowmap(i) << ' ';
    }
    std::cout << '\n';
    std::cout << "Checking matrix validity...\n";
    for(offset_t i = 0; i < rowmap(numRows); i++)
    {
      if(colinds(i) < 0 || colinds(i) >= numRows)
      {
        std::cout << "Entry " << i << " (" << colinds(i) << ") is invalid.\n";
      }
    }
    */
    //std::cout << "Working on matrix with " << numRows << " rows and " << rowmap(numRows) << " entries.\n";
    //need to know maximum degree to allocate scratch space for threads
    auto maxDeg = find_max_degree();
    /*
    std::cout << "Max degree of graph: " << maxDeg << '\n';
    std::cout << "Row counts: ";
    for(nnz_lno_t i = 0; i < numRows; i++)
    {
      std::cout << rowmap(i + 1) - rowmap(i) << ' ';
    }
    std::cout << "\n\n\n";
    */
    //place these two magic values are at the top of nnz_lno_t's range
    const nnz_lno_t LNO_MAX = Kokkos::ArithTraits<nnz_lno_t>::max();
    const nnz_lno_t NOT_VISITED = LNO_MAX;
    const nnz_lno_t QUEUED = NOT_VISITED - 1;
    //view for storing the visit timestamps
    result_view_t visit("BFS visited nodes", numRows);
    Kokkos::parallel_for(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(size_type i) {visit(i) = NOT_VISITED;});
    //visitCounter atomically counts timestamps
    single_view_t visitCounter("BFS visit counter");
    //the visit queue
    //one of q1,q2 is active at a time and holds the nodes to process in next BFS level
    //elements which are LNO_MAX are just placeholders (nothing to process)
    size_type nthreads = MyExecSpace::concurrency();
    if(nthreads > 64)
      nthreads = 64;
    #ifdef KOKKOS_ENABLE_CUDA
    if(std::is_same<MyExecSpace, Kokkos::Cuda>::value)
    {
      nthreads = 256;
    }
    #endif
    Kokkos::View<nnz_lno_t**, MyTempMemorySpace, Kokkos::MemoryTraits<0>> workQueue("BFS queue (double buffered)", 2, numRows);
    Kokkos::View<nnz_lno_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0>> updateThreads("Thread IDs adding nodes to queue", numRows);
    Kokkos::parallel_for(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(size_type i) {updateThreads(i) = LNO_MAX;});
    perm_view_t updateThreadCounts("Number of nodes to queue on each thread", nthreads);
    single_view_t updateDirty("Whether queue update pass must run");
    single_view_t activeQSize("BFS active level queue size");
    single_view_t nextQSize("BFS next level queue size");
    //TODO: place this in shared. Check for allocation failure, fall back to global?
    //std::cout << "Allocating " << nthreads * maxDeg << " total elements of scratch to share among " << nthreads << ".\n";
    Kokkos::View<nnz_lno_t**, MyTempMemorySpace, Kokkos::MemoryTraits<0u>> scratch("Scratch buffer shared by threads", nthreads, maxDeg * 4);
    //std::cout << "Launching outer team policy with " << nthreads << " threads per team.\n";
    Kokkos::parallel_for(team_policy_t(1, nthreads),
    KOKKOS_LAMBDA(const team_member_t mem)
    {
      nnz_lno_t tid = mem.team_rank();
      auto neighborList = Kokkos::subview(scratch, tid, Kokkos::make_pair(0, maxDeg));
      auto neighborListAux = Kokkos::subview(scratch, tid, Kokkos::make_pair(maxDeg, maxDeg * 2));
      auto degreeList = Kokkos::subview(scratch, tid, Kokkos::make_pair(maxDeg * 2, maxDeg * 3));
      auto degreeListAux = Kokkos::subview(scratch, tid, Kokkos::make_pair(maxDeg * 3, maxDeg * 4));
      nnz_lno_t nextQSizeLocal = 0;
      int active = 0;
      int next = 1;
      Kokkos::single(Kokkos::PerTeam(mem),
      KOKKOS_LAMBDA()
      {
        workQueue(active, 0) = start;
        activeQSize() = 1;
        nextQSize() = 0;
      });
      mem.team_barrier();
      //do until every node has been visited and labeled
      while(visitCounter() < numRows)
      {
        mem.team_barrier();
        auto qSize = activeQSize();
        for(size_type workSet = 0; workSet < qSize; workSet += nthreads)
        {
          mem.team_barrier();
          nextQSizeLocal = nextQSize();
          nnz_lno_t neiCount = 0;
          nnz_lno_t process = LNO_MAX;
          //is thread tid visiting a node this iteration?
          bool busy = workSet + tid < qSize;
          if(busy)
          {
            //get pointer to thread-local scratch space, which has size maxDeg
            //the node to process
            process = workQueue(active, workSet + tid);
            offset_t rowStart = rowmap(process);
            offset_t rowEnd = rowmap(process + 1);
            //build a list of all non-visited neighbors
            for(offset_t j = rowStart; j < rowEnd; j++)
            {
              nnz_lno_t col = colinds(j);
              if(visit(col) == NOT_VISITED && col != process)
              {
                neighborList(neiCount) = col;
                degreeList(neiCount) = rowEnd - rowStart;
                neiCount++;
              }
            }
            //this sort will sort neighborList according to degree
            radixSortKeysAndValues<size_type, nnz_lno_t, nnz_lno_t, size_type>(degreeList.data(), degreeListAux.data(), neighborList.data(), neighborListAux.data(), neiCount);
          }
          //must do at least one pass of tracking update threads
          Kokkos::single(Kokkos::PerTeam(mem),
          KOKKOS_LAMBDA()
          {
            updateDirty() = 1;
          });
          mem.team_barrier();
          while(updateDirty())
          {
            mem.team_barrier();
            Kokkos::single(Kokkos::PerTeam(mem),
            KOKKOS_LAMBDA()
            {
              updateDirty() = 0;
            });
            mem.team_barrier();
            //go through rows in thread-local scratch
            //replace updateThreads(row) with tid, if tid is lower
            for(size_type i = 0; i < neiCount; i++)
            {
              nnz_lno_t prevThread = updateThreads(neighborList(i));
              if(prevThread > tid)
              {
                //swap in tid atomically. If this fails there is a conflict and
                //will need to go through loop again
                if(prevThread != Kokkos::atomic_exchange<nnz_lno_t>(&updateThreads(neighborList(i)), tid))
                {
                  Kokkos::atomic_increment(&updateDirty());
                }
              }
            }
            mem.team_barrier();
          }
          //each thread counts the number of entries it will actually add to nextQ
          size_type numToQueue = 0;
          for(size_type i = 0; i < neiCount; i++)
          {
            if(updateThreads(neighborList(i)) == tid)
            {
              numToQueue++;
            }
          }
          updateThreadCounts(tid) = numToQueue;
          mem.team_barrier();
          size_type queueUpdateOffset = 0;
          for(size_type i = 0; i < tid; i++)
          {
            queueUpdateOffset += updateThreadCounts(i);
          }
          //write out all queue updates in parallel (only for threads that worked this iteration)
          if(busy)
          {
            size_type nextQueueIter = 0;
            for(size_type i = 0; i < neiCount; i++)
            {
              nnz_lno_t toQueue = neighborList(i);
              if(updateThreads(toQueue) == tid)
              {
                visit(toQueue) = QUEUED;
                workQueue(next, nextQSizeLocal + queueUpdateOffset + nextQueueIter) = toQueue;
                nextQueueIter++;
              }
            }
            //apply correct label to process
            visit(process) = visitCounter() + tid;
          }
          mem.team_barrier();
          Kokkos::single(Kokkos::PerTeam(mem),
          KOKKOS_LAMBDA()
          {
            if(workSet + nthreads > qSize)
              visitCounter() += qSize - workSet;
            else
              visitCounter() += nthreads;
            //update queue size by the number of nodes added this iteration
            size_type totalQueueAdded = 0;
            for(size_type i = 0; i < nthreads; i++)
            {
              totalQueueAdded += updateThreadCounts(i);
            }
            nextQSize() += totalQueueAdded;
          });
        }
        //(thread-local) swap queue buffers
        {
          active = next;
          next = 1 - next;
        }
        mem.team_barrier();
        Kokkos::single(Kokkos::PerTeam(mem),
        KOKKOS_LAMBDA()
        {
          activeQSize() = nextQSize();
          nextQSize() = 0;
          if(visitCounter() < numRows && activeQSize() == 0)
          {
            //Some nodes are unreachable (graph not connected)
            //Find an unvisited node to resume BFS
            for(nnz_lno_t search = numRows - 1; search >= 0; search--)
            {
              if(visit(search) == NOT_VISITED)
              {
                workQueue(active, 0) = search;
                activeQSize() = 1;
                visit(search) = QUEUED;
                std::cout << "WARNING: graph not connected. Resuming BFS from node " << search << std::endl;
                break;
              }
            }
          }
        });
        mem.team_barrier();
      }
    });
    return visit;
  }

  //Find a peripheral node, either
  nnz_lno_t find_peripheral(PeripheralMode mode)
  {
    //if(mode == MIN_DEGREE)
    //{
      ValLoc v;
      v.val = numRows;
      Kokkos::parallel_reduce(range_policy_t(0, numRows),
        KOKKOS_LAMBDA(size_type i, ValLoc& lminloc)
        {
          offset_t nnzInRow = rowmap(i + 1) - rowmap(i);
          if((nnz_lno_t) nnzInRow < lminloc.val)
          {
            lminloc.val = nnzInRow;
            lminloc.loc = i;
          }
        }, MinLocReducer(v));
      return v.loc;
    /*
    }
    else
    {
      //BFS from node 0, then BFS again from the most distant node
      nnz_lno_t mostDistant;
      {
        auto visitOrder1 = parallel_bfs(0);
        ValLoc v;
        v.val = 0;
        Kokkos::parallel_reduce(range_policy_t(0, numRows),
          KOKKOS_LAMBDA(size_type i, ValLoc& lmaxloc)
          {
            if(visitOrder1(i) > lmaxloc.val)
            {
              lmaxloc.val = visitOrder1(i);
              lmaxloc.loc = i;
            }
          }, MaxLocReducer(v));
        mostDistant = v.loc;
      }
      ValLoc v;
      v.val = 0;
      auto visitOrder2 = parallel_bfs(mostDistant);
      Kokkos::parallel_reduce(range_policy_t(0, numRows),
        KOKKOS_LAMBDA(size_type i, ValLoc& lmaxloc)
        {
          if(visitOrder2(i) > lmaxloc.val)
          {
            lmaxloc.val = visitOrder2(i);
            lmaxloc.loc = i;
          }
        }, MaxLocReducer(v));
      return v.loc;
    }
    */
    return 0;
  }

  result_view_t rcm()
  {
    //find a peripheral node
    //TODO: make mode configurable, but keep MIN_DEGREE the default
    nnz_lno_t periph = find_peripheral(MIN_DEGREE);
    //run Cuthill-McKee BFS from periph
    result_view_t visit = parallel_rcm_bfs(periph);
    //reverse the visit order (for "reverse" C-M)
    Kokkos::parallel_for(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(size_type i)
      {
        visit(i) = numRows - 1 - visit(i);
      });
    return visit;
  }
};

}}  //KokkosSparse::Impl

#endif


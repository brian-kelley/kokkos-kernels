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

  //radix sort keys according to their corresponding values ascending.
  //keys are NOT preserved since the use of this in RCM doesn't care about degree after sorting
  template<typename size_type, typename KeyType, typename ValueType, typename IndexType, typename member_t>
  KOKKOS_INLINE_FUNCTION void
  radixSortKeysAndValues(KeyType* keys, KeyType* keysAux, ValueType* values, ValueType* valuesAux, IndexType n, member_t& mem)
  {
    if(n <= 1)
      return;
    //sort 4 bits at a time
    KeyType mask = 0xF;
    bool inAux = false;
    //maskPos counts the low bit index of mask (0, 4, 8, ...)
    IndexType maskPos = 0;
    IndexType sortBits = 0;
    KeyType minKey = Kokkos::ArithTraits<KeyType>::max();
    KeyType maxKey = 0;
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(mem, n),
    KOKKOS_LAMBDA(size_type i, KeyType& lminkey)
    {
      if(keys[i] < lminkey)
        lminkey = keys[i];
    }, Kokkos::Min<KeyType>(minKey));
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(mem, n),
    KOKKOS_LAMBDA(size_type i, KeyType& lmaxkey)
    {
      if(keys[i] > lmaxkey)
        lmaxkey = keys[i];
    }, Kokkos::Max<KeyType>(maxKey));
    //subtract a bias of minKey so that key range starts at 0
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(mem, n),
    KOKKOS_LAMBDA(size_type i)
    {
      keys[i] -= minKey;
    });
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
      IndexType offset[17];
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
      offset[0] = 0;
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
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(mem, n),
      KOKKOS_LAMBDA(size_type i)
      {
        values[i] = valuesAux[i];
      });
    }
  }

  /*
   * -Nodes in a frontier must be in ascending order of predecessor, and within that by degree
   * -Nodes must be labeled in same order as they appear in queue (easy)
   * -General idea:
   *    -ideally, only one barrier per frontier
   *    -takes one node to work on and atomically increments queue tail
   *    -gets list of non-labeled neighbors (inexact) and sorts by degree (exact)
   *    -want to visit nodes in each frontier in order of predecessor, then by degree
   */

  //breadth-first search, producing a reverse Cuthill-McKee ordering
  result_view_t serial_rcm_bfs(nnz_lno_t start)
  {
    //need to know maximum degree to allocate scratch space for threads
    auto maxDeg = find_max_degree();
    //place these two magic values are at the top of nnz_lno_t's range
    const nnz_lno_t LNO_MAX = Kokkos::ArithTraits<nnz_lno_t>::max();
    const nnz_lno_t NOT_VISITED = LNO_MAX;
    const nnz_lno_t QUEUED = NOT_VISITED - 1;
    //view for storing the visit timestamps
    result_view_t visit("BFS visited nodes", numRows);
    Kokkos::parallel_for(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(size_type i) {visit(i) = NOT_VISITED;});
    //the visit queue
    //one of q1,q2 is active at a time and holds the nodes to process in next BFS level
    //elements which are LNO_MAX are just placeholders (nothing to process)
    size_type nthreads = 1;
    Kokkos::View<nnz_lno_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0>> workQueue("BFS queue", numRows);
    Kokkos::View<nnz_lno_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0u>> scratch("Scratch buffer", maxDeg * 4);
    Kokkos::parallel_for(team_policy_t(1, 1),
    KOKKOS_LAMBDA(const team_member_t mem)
    {
      nnz_lno_t qHead = 1;
      nnz_lno_t qTail = 0;
      workQueue(0) = start;
      nnz_lno_t visitCounter = 0;
      auto neighborList = Kokkos::subview(scratch, Kokkos::make_pair(0, maxDeg));
      auto neighborListAux = Kokkos::subview(scratch, Kokkos::make_pair(maxDeg, maxDeg * 2));
      auto degreeList = Kokkos::subview(scratch, Kokkos::make_pair(maxDeg * 2, maxDeg * 3));
      auto degreeListAux = Kokkos::subview(scratch, Kokkos::make_pair(maxDeg * 3, maxDeg * 4));
      while(visitCounter < numRows)
      {
        //get pointer to thread-local scratch space, which has size maxDeg
        //the node to process
        nnz_lno_t process = workQueue(qTail);
        qTail++;
        offset_t rowStart = rowmap(process);
        offset_t rowEnd = rowmap(process + 1);
        size_type neiCount = 0;
        //build a list of all non-visited neighbors
        for(offset_t i = rowStart; i < rowEnd; i++)
        {
          nnz_lno_t col = colinds(i);
          if(visit(col) == NOT_VISITED && col != process)
          {
            neighborList(neiCount) = col;
            degreeList(neiCount) = rowEnd - rowStart;
            neiCount++;
          }
        }
        //this sort will sort neighborList according to degree
        radixSortKeysAndValues<size_type, nnz_lno_t, nnz_lno_t, size_type>
          (degreeList.data(), degreeListAux.data(), neighborList.data(), neighborListAux.data(), neiCount, mem);
        for(offset_t i = 0; i < neiCount; i++)
        {
          visit(neighborList(i)) = QUEUED;
          workQueue(qHead) = neighborList(i);
          qHead++;
        }
        visit(process) = numRows - 1 - visitCounter;
        visitCounter++;
        if(visitCounter < numRows && qTail == qHead)
        {
          //Some nodes are unreachable from start (graph not connected)
          //Find an unvisited node to resume BFS
          for(nnz_lno_t search = numRows - 1; search >= 0; search--)
          {
            if(visit(search) == NOT_VISITED)
            {
              workQueue(qHead) = search;
              qHead++;
              visit(search) = QUEUED;
              std::cout << "WARNING: graph not connected. Resuming BFS from node " << search << std::endl;
              break;
            }
          }
        }
      }
    });
    return visit;
  }

/*
  //breadth-first search, producing a reverse Cuthill-McKee ordering
  result_view_t parallel_rcm_bfs(nnz_lno_t start)
  {
    //need to know maximum degree to allocate scratch space for threads
    auto maxDeg = find_max_degree();
    //place these two magic values are at the top of nnz_lno_t's range
    const nnz_lno_t LNO_MAX = Kokkos::ArithTraits<nnz_lno_t>::max();
    const nnz_lno_t NOT_VISITED = LNO_MAX;
    const nnz_lno_t QUEUED = NOT_VISITED - 1;
    //view for storing the visit timestamps
    result_view_t visit("BFS visited nodes", numRows);
    Kokkos::parallel_for(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(size_type i) {visit(i) = NOT_VISITED;});
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
    perm_view_t updateThreadCounts("Number of nodes to queue on each thread", nthreads);
    single_view_t activeQSize("BFS active level queue size");
    single_view_t nextQSize("BFS next level queue size");
    //TODO: place this in shared. Check for allocation failure, fall back to global?
    Kokkos::View<nnz_lno_t**, MyTempMemorySpace, Kokkos::MemoryTraits<0u>> scratch("Scratch buffer shared by threads", nthreads, maxDeg * 4);
    Kokkos::parallel_for(team_policy_t(1, nthreads),
    KOKKOS_LAMBDA(const team_member_t mem)
    {
      nnz_lno_t tid = mem.team_rank();
      auto neighborList = Kokkos::subview(scratch, tid, Kokkos::make_pair(0, maxDeg));
      auto neighborListAux = Kokkos::subview(scratch, tid, Kokkos::make_pair(maxDeg, maxDeg * 2));
      auto degreeList = Kokkos::subview(scratch, tid, Kokkos::make_pair(maxDeg * 2, maxDeg * 3));
      auto degreeListAux = Kokkos::subview(scratch, tid, Kokkos::make_pair(maxDeg * 3, maxDeg * 4));
      //a thread-local copy of nextQSize
      nnz_lno_t nextQSizeLocal = 0;
      //active and next indicate which buffer in workQueue holds the nodes in current/next frontiers, respectively
      //active, next and visitCounter are local but kept consistent across threads
      int active = 0;
      int next = 1;
      nnz_lno_t visitCounter = 0;
      Kokkos::single(Kokkos::PerTeam(mem),
      KOKKOS_LAMBDA()
      {
        workQueue(active, 0) = start;
        activeQSize() = 1;
        //note: nextQSize() is automatically 0
      });
      //do until every node has been visited and labeled
      while(visitCounter < numRows)
      {
        mem.team_barrier();
        auto qSize = activeQSize();
        for(size_type workSet = 0; workSet < qSize; workSet += nthreads)
        {
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
            radixSortKeysAndValues<size_type, nnz_lno_t, nnz_lno_t, size_type>
              (degreeList.data(), degreeListAux.data(), neighborList.data(), neighborListAux.data(), neiCount, mem);
            for(size_type i = 0; i < neiCount; i++)
            {
              nnz_lno_t prevThread = Kokkos::volatile_load(&visit(neighborList(i)));
              while(prevThread > tid)
              {
                if(Kokkos::atomic_compare_exchange_strong<nnz_lno_t>(&visit(neighborList(i)), prevThread, tid))
                {
                  //update succeeded, this thread done
                  break;
                }
                else
                {
                  prevThread = Kokkos::volatile_load(&visit(neighborList(i)));
                }
              }
            }
          }
          //barrier so that all threads have finished setting visit entries
          mem.team_barrier();
          //here is a good place to update nextQSizeLocal (between two barriers)
          nextQSizeLocal = nextQSize();
          //each thread counts the number of entries it will actually add to nextQ
          size_type numToQueue = 0;
          for(size_type i = 0; i < neiCount; i++)
          {
            if(visit(neighborList(i)) == tid)
            {
              numToQueue++;
            }
          }
          updateThreadCounts(tid) = numToQueue;
          mem.team_barrier();
          if(busy)
          {
            size_type queueUpdateOffset = 0;
            for(size_type i = 0; i < tid; i++)
            {
              queueUpdateOffset += updateThreadCounts(i);
            }
            //write out all queue updates in parallel (only for threads that worked this iteration)
            size_type nextQueueIter = 0;
            for(size_type i = 0; i < neiCount; i++)
            {
              nnz_lno_t toQueue = neighborList(i);
              if(visit(toQueue) == tid)
              {
                visit(toQueue) = QUEUED;
                workQueue(next, nextQSizeLocal + queueUpdateOffset + nextQueueIter) = toQueue;
                nextQueueIter++;
              }
            }
            //apply correct label to process (reversed to get RCM)
            visit(process) = (numRows - 1) - (visitCounter + workSet + tid);
          }
          Kokkos::atomic_fetch_add(&nextQSize(), numToQueue);
          //need this barrier since visit is read during next iteration
          mem.team_barrier();
        }
        //swap queue buffers
        active = next;
        next = 1 - next;
        //all threads have a consistent value of qSize here.
        //update visitCounter in preparation for next frontier
        visitCounter += qSize;
        mem.team_barrier();
        Kokkos::single(Kokkos::PerTeam(mem),
        KOKKOS_LAMBDA()
        {
          //know exactly how many vertices were labeled in the last frontier
          activeQSize() = nextQSize();
          nextQSize() = 0;
          if(visitCounter < numRows && activeQSize() == 0)
          {
            //Some nodes are unreachable from start (graph not connected)
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
      }
    });
    return visit;
  }
  */

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
    return serial_rcm_bfs(periph);
  }
};

}}  //KokkosSparse::Impl

#endif


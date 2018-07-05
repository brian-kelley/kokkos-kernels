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

  typedef Kokkos::View<nnz_lno_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0>> nnz_view_t;
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
  radixSortKeysAndValues(KeyType* keys, KeyType* keysAux, ValueType* values, ValueType* valuesAux, IndexType n, const member_t& mem)
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
    //apply a bias so that key range always starts at 0
    //also invert key values here for a descending sort
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
    //move keys/values back from aux if they are currently in aux,
    //and remove bias
    if(inAux)
    {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(mem, n),
      KOKKOS_LAMBDA(size_type i)
      {
        //TODO: when everything works, is safe to remove next line
        //since keys (BFS visit scores) will never be needed again
        keys[i] = keysAux[i];
        values[i] = valuesAux[i];
      });
    }
  }

  //breadth-first search, producing a reverse Cuthill-McKee ordering
  nnz_view_t serial_rcm(nnz_lno_t start)
  {
    //need to know maximum degree to allocate scratch space for threads
    auto maxDeg = find_max_degree();
    //place these two magic values are at the top of nnz_lno_t's range
    const nnz_lno_t LNO_MAX = Kokkos::ArithTraits<nnz_lno_t>::max();
    const nnz_lno_t NOT_VISITED = LNO_MAX;
    const nnz_lno_t QUEUED = NOT_VISITED - 1;
    //view for storing the visit timestamps
    nnz_view_t visit("BFS visited nodes", numRows);
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
              break;
            }
          }
        }
      }
    });
    return visit;
  }

  //parallel breadth-first search, producing level structure in (xadj, adj) form:
  //xadj(level) gives index in adj where level begins
  //also return the total number of levels
  nnz_lno_t parallel_bfs(nnz_lno_t start, nnz_view_t& xadj, nnz_view_t& adj, nnz_lno_t& maxDeg, size_type nthreads)
  {
    //need to know maximum degree to allocate scratch space for threads
    maxDeg = find_max_degree();
    //place these two magic values are at the top of nnz_lno_t's range
    const nnz_lno_t LNO_MAX = Kokkos::ArithTraits<nnz_lno_t>::max();
    const nnz_lno_t NOT_VISITED = LNO_MAX;
    const nnz_lno_t QUEUED = NOT_VISITED - 1;
    //view for storing the visit timestamps
    nnz_view_t visit("BFS visited nodes", numRows);
    Kokkos::parallel_for(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(size_type i) {visit(i) = NOT_VISITED;});
    //the visit queue
    //one of q1,q2 is active at a time and holds the nodes to process in next BFS level
    //elements which are LNO_MAX are just placeholders (nothing to process)
    Kokkos::View<nnz_lno_t**, MyTempMemorySpace, Kokkos::MemoryTraits<0>> workQueue("BFS queue (double buffered)", 2, numRows);
    nnz_view_t threadNeighborCounts("Number of nodes to queue on each thread", nthreads);
    single_view_t numLevels("# of BFS levels");
    Kokkos::View<nnz_lno_t**, MyTempMemorySpace, Kokkos::MemoryTraits<0u>> scratch("Scratch buffer shared by threads", nthreads, maxDeg);
    Kokkos::parallel_for(team_policy_t(1, nthreads),
    KOKKOS_LAMBDA(const team_member_t mem)
    {
      nnz_lno_t tid = mem.team_rank();
      auto neighborList = Kokkos::subview(scratch, tid, Kokkos::ALL());
      //active and next indicate which buffer in workQueue holds the nodes in current/next frontiers, respectively
      //active, next and visitCounter are thread-local, but always kept consistent across threads
      int active = 0;
      int next = 1;
      nnz_lno_t visitCounter = 0;
      Kokkos::single(Kokkos::PerTeam(mem),
      KOKKOS_LAMBDA()
      {
        workQueue(active, 0) = start;
        visit(start) = QUEUED;
      });
      nnz_lno_t activeQSize = 1;
      nnz_lno_t nextQSize = 0;
      //KK create_reverse_map() expects incoming values to start at 1
      nnz_lno_t level = 1;
      //do this until all nodes have been visited and added to a level
      while(visitCounter < numRows)
      {
        mem.team_barrier();
        //each thread works on a contiguous block of nodes in queue (for locality)
        //compute in size_t to avoid possible 32-bit overflow
        size_type workStart = (size_t) tid * activeQSize / nthreads;
        size_type workEnd = (size_t) (tid + 1) * activeQSize / nthreads;
        //the maximum work batch size (among all threads)
        //the following loop contains barriers so all threads must iterate same # of times
        size_type maxBatch = (activeQSize + nthreads - 1) / nthreads;
        for(size_type loop = 0; loop < maxBatch; loop++)
        {
          //this thread may not actually have anything to work on (if nthreads doesn't divide qSize)
          bool busy = loop < workEnd - workStart;
          nnz_lno_t neiCount = 0;
          nnz_lno_t process = LNO_MAX;
          if(busy)
          {
            process = workQueue(active, workStart + loop);
            offset_t rowStart = rowmap(process);
            offset_t rowEnd = rowmap(process + 1);
            //build a list of all non-visited neighbors
            for(offset_t j = rowStart; j < rowEnd; j++)
            {
              nnz_lno_t col = colinds(j);
              //use atomic here to guarantee neighbors are added to neighborList exactly once
              if(Kokkos::atomic_compare_exchange_strong<nnz_lno_t>(&visit(col), NOT_VISITED, QUEUED))
              {
                //this thread is the first to see that col needs to be queued
                neighborList(neiCount) = col;
                neiCount++;
              }
            }
          }
          threadNeighborCounts(tid) = neiCount;
          mem.team_barrier();
          size_type queueUpdateOffset = 0;
          for(size_type i = 0; i < tid; i++)
          {
            queueUpdateOffset += threadNeighborCounts(i);
          }
          //write out all updates to next queue in parallel
          if(busy)
          {
            size_type nextQueueIter = 0;
            for(size_type i = 0; i < neiCount; i++)
            {
              nnz_lno_t toQueue = neighborList(i);
              visit(toQueue) = QUEUED;
              workQueue(next, nextQSize + queueUpdateOffset + nextQueueIter) = toQueue;
              nextQueueIter++;
            }
            //assign level to to process
            visit(process) = level;
          }
          size_type totalAdded = 0;
          for(nnz_lno_t i = 0; i < nthreads; i++)
          {
            totalAdded += threadNeighborCounts(i);
          }
          nextQSize += totalAdded;
          mem.team_barrier();
        }
        //swap queue buffers
        active = next;
        next = 1 - next;
        //all threads have a consistent value of qSize here.
        //update visitCounter in preparation for next frontier
        visitCounter += activeQSize;
        activeQSize = nextQSize;
        nextQSize = 0;
        if(visitCounter < numRows && activeQSize == 0)
        {
          Kokkos::single(Kokkos::PerTeam(mem),
          KOKKOS_LAMBDA()
          {
            //Some nodes are unreachable from start (graph not connected)
            //Find an unvisited node to resume BFS
            for(nnz_lno_t search = numRows - 1; search >= 0; search--)
            {
              if(visit(search) == NOT_VISITED)
              {
                workQueue(active, 0) = search;
                visit(search) = QUEUED;
                break;
              }
            }
          });
          activeQSize = 1;
        }
        level++;
      }
      Kokkos::single(Kokkos::PerTeam(mem),
      KOKKOS_LAMBDA()
      {
        numLevels() = level - 1;
      });
    });
    //now that level structure has been computed, construct xadj/adj
    KokkosKernels::Impl::create_reverse_map<nnz_view_t, nnz_view_t, MyExecSpace>
      (numRows, numLevels(), visit, xadj, adj);
    return numLevels();
  }

  //breadth-first search, producing a reverse Cuthill-McKee ordering
  nnz_view_t parallel_rcm(nnz_lno_t start)
  {
    size_type nthreads = MyExecSpace::concurrency();
    if(nthreads > 64)
      nthreads = 64;
    #ifdef KOKKOS_ENABLE_CUDA
    if(std::is_same<MyExecSpace, Kokkos::Cuda>::value)
    {
      nthreads = 256;
    }
    #endif
    nnz_view_t xadj, adj;
    nnz_lno_t maxDegree = 0;
    //parallel_bfs will compute maxDegree
    auto numLevels = parallel_bfs(start, xadj, adj, maxDegree, nthreads);
    nnz_lno_t maxLevelSize = 0;
    Kokkos::parallel_reduce(range_policy_t(0, numLevels),
    KOKKOS_LAMBDA(size_type i, nnz_lno_t& lmax)
    {
      nnz_lno_t thisLevelSize = xadj(i + 1) - xadj(i);
      if(thisLevelSize > lmax)
        lmax = thisLevelSize;
    }, Kokkos::Max<nnz_lno_t>(maxLevelSize));
    //visit (to be returned) contains the RCM numberings of each row
    nnz_view_t visit("RCM labels", numRows);
    //Populate visit wth LNO_MAX so that the "min-labeled neighbor"
    //is always a node in the previous level
    const nnz_lno_t LNO_MAX = Kokkos::ArithTraits<nnz_lno_t>::max();
    Kokkos::parallel_for(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(size_type i) {visit(i) = LNO_MAX;});
    //the "score" of a node is a single value that provides an ordering equivalent
    //to sorting by min predecessor and then by min degree
    //reduce nthreads to be a power of 2
    Kokkos::View<offset_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0u>> scores("RCM scores for sorting", maxLevelSize);
    Kokkos::View<offset_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0u>> scoresAux("RCM scores for sorting (radix sort aux)", maxLevelSize);
    Kokkos::View<nnz_lno_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0u>> adjAux("RCM scores for sorting (radix sort aux)", maxLevelSize);
    Kokkos::parallel_for(team_policy_t(1, nthreads),
    KOKKOS_LAMBDA(const team_member_t mem)
    {
      auto tid = mem.team_rank();
      nnz_lno_t visitCounter = 0;
      for(nnz_lno_t level = 0; level < numLevels; level++)
      {
        //iterate over vertices in this level (parallel) and compute
        //min predecessors (minimum-labeled vertices from previous level)
        nnz_lno_t levelOffset = xadj(level);
        nnz_lno_t levelSize = xadj(level + 1) - levelOffset;
        //compute as size_t to avoid overflow 
        size_type workStart = (size_t) tid * levelSize / nthreads;
        size_type workEnd = (size_t) (tid + 1) * levelSize / nthreads;
        for(size_type i = workStart; i < workEnd; i++)
        {
          nnz_lno_t process = adj(levelOffset + i);
          nnz_lno_t minNeighbor = LNO_MAX;
          offset_t rowStart = rowmap(process);
          offset_t rowEnd = rowmap(process + 1);
          for(offset_t j = rowStart; j < rowEnd; j++)
          {
            nnz_lno_t neighbor = colinds(j);
            nnz_lno_t neighborVisit = visit(neighbor);
            if(neighborVisit < minNeighbor)
              minNeighbor = neighborVisit;
          }
          scores(i) = ((offset_t) minNeighbor * (maxDegree + 1)) + (rowmap(process + 1) - rowmap(process));
        }
        mem.team_barrier();
        Kokkos::single(Kokkos::PerTeam(mem),
        KOKKOS_LAMBDA()
        {
          radixSortKeysAndValues<size_type, offset_t, nnz_lno_t, nnz_lno_t, team_member_t>
            (scores.data(), scoresAux.data(), adj.data() + levelOffset, adjAux.data(), levelSize, mem);
        });
        mem.team_barrier();
        //label all vertices (which are now in label order within their level)
        for(size_type i = workStart; i < workEnd; i++)
        {
          nnz_lno_t process = adj(levelOffset + i);
          //visit counter increases with levels, so flip the range for the "reverse" in RCM
          visit(process) = visitCounter + i;
        }
        visitCounter += levelSize;
      }
    });
    //reverse the visit order (for the 'R' in RCM)
    Kokkos::parallel_for(range_policy_t(0, numRows),
    KOKKOS_LAMBDA(size_type i)
    {
      visit(i) = numRows - 1 - visit(i);
    });
    return visit;
  }

  //Find a peripheral node, either
  nnz_lno_t find_peripheral()
  {
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
  }

  nnz_view_t rcm()
  {
    nnz_lno_t periph = find_peripheral();
    //run Cuthill-McKee BFS from periph
    return parallel_rcm(periph);
    //return serial_rcm(periph);
  }
};

}}  //KokkosSparse::Impl

#endif


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

template <typename HandleType, typename lno_row_view_t, typename lno_nnz_view_t, typename perm_view_t>
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

  typedef Kokkos::View<nnz_lno_t, MyTempMemorySpace, Kokkos::MemoryManaged> single_view_t;

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

  //compute a breadth-first search from start node
  //returns a view containing the visit timestamps for each node (starting at 2)
  //timestamp = 0 means not visited yet
  //timestamp = 1 means node is queued for visit in next iteration
  //at return, visit[start] = 2, visit[i] = visit time of node i
  //this implementation is parallel but nondeterministic
  non_const_lno_nnz_view_t parallel_bfs(nnz_lno_t start)
  {
    const nnz_lno_t NOT_VISITED = 0;
    const nnz_lno_t QUEUED = 1;
    const nnz_lno_t START = 2;
    //view for storing the visit timestamps
    perm_view_t visit("BFS visited nodes", numRows);
    //visitCounter atomically counts timestamps
    single_view_t visitCounter("BFS visit counter (atomic)");
    //the visit queue
    //will process the elements in [qStart, qEnd) in parallel during each sweep
    //the queue doesn't need to be circular since each node is visited exactly once
    perm_view_t q("BFS queue", numRows);
    single_view_t qStart("BFS frontier start index (last in queue)");
    single_view_t qEnd("BFS frontier end index (next in queue)");
    Kokkos::parallel_for(team_policy_t(1, Kokkos::AUTO()),
      KOKKOS_LAMBDA(team_member_t mem)
      {
        //initialize the frontier as just the starting node
        Kokkos::single(Kokkos::PerThread(mem),
          KOKKOS_LAMBDA()
          {
            visitCounter() = START;
            visit(start) = QUEUED;
            q(0) = start;
            qStart() = 0;
            qEnd() = 1;
          });
        auto tid = mem.team_rank();
        //all threads work until every node has been labeled
        while(qStart() != qEnd())
        {
          //loop over the frontier, giving each thread one node to process
          size_type workStart = qStart();
          size_type workEnd = qEnd();
          mem.team_barrier();
          qStart() = qEnd();
          for(size_type teamIndex = workStart; teamIndex < workEnd; teamIndex += mem.team_size())
          {
            if(teamIndex + tid < workEnd)
            {
              size_type i = teamIndex + tid;
              //the node to process
              nnz_lno_t process = q(i);
              size_t rowStart = rowmap(process);
              size_t rowEnd = rowmap(process + 1);
              //loop over neighbors, enqueing all which are NOT_VISITED
              for(size_t j = rowStart; j < rowEnd; j++)
              {
                nnz_lno_t col = colinds(j);
                if(Kokkos::atomic_compare_exchange_strong<nnz_lno_t>(&visit(col), NOT_VISITED, QUEUED))
                {
                  //compare-exchange passed, so append col to queue (for next iteration)
                  q(Kokkos::atomic_fetch_add(&qEnd(), 1)) = col;
                }
              }
              //label current node with the next timestamp
              visit(process) = Kokkos::atomic_fetch_add(&visitCounter(), 1);
            }
          }
          mem.team_barrier();
        }
      });
    return visit;
  }

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
      }, maxDeg);
    return maxDeg;
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

  //breadth-first search, producing a Cuthill-McKee ordering
  //nodes are labeled in the exact order they would be in a serial BFS,
  //and neighbors are visited in ascending order of degree
  perm_view_t parallel_rcm_bfs(nnz_lno_t start)
  {
    //std::cout << "In rcm().\n";
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
    perm_view_t visit("BFS visited nodes", numRows);
    Kokkos::parallel_for(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(size_type i) {visit(i) = NOT_VISITED;});
    //visitCounter atomically counts timestamps
    single_view_t visitCounter("BFS visit counter");
    //the visit queue
    //one of q1,q2 is active at a time and holds the nodes to process in next BFS level
    //elements which are LNO_MAX are just placeholders (nothing to process)
    size_type nthreads = MyExecSpace::concurrency();
    #ifdef KOKKOS_ENABLE_CUDA
    if(std::is_same<MyExecSpace, Kokkos::Cuda>::value)
    {
      nthreads = 256;
    }
    #endif
    perm_view_t q1("BFS queue (buf 1)", numRows);
    perm_view_t q2("BFS queue (buf 2)", numRows);
    perm_view_t updateThreads("Thread IDs adding nodes to queue", numRows);
    perm_view_t updateThreadCounts("Number of nodes to queue on each thread", nthreads);
    single_view_t updateDirty("Whether queue update pass must run");
    single_view_t activeQSize("BFS active level queue size");
    single_view_t nextQSize("BFS next level queue size");
    //TODO: place this in shared. Check for allocation failure, fall back to global?
    std::cout << "Allocating " << nthreads * maxDeg << " total elements of scratch to share among " << nthreads << ".\n";
    perm_view_t scratchSpace("Scratch buffer shared by threads", nthreads * maxDeg);
    std::cout << "Launching outer team policy with " << nthreads << " threads per team.\n";
    Kokkos::parallel_for(team_policy_t(1, nthreads),
    KOKKOS_LAMBDA(const team_member_t& mem)
    {
      //swap these buffers after each level
      nnz_lno_t* teamScratch = scratchSpace.data();
      //all threads work until every node has been labeled (queue empty)
      //std::cout << "Launching inner thread policy.\n";
      Kokkos::parallel_for(Kokkos::TeamThreadRange(mem, nthreads),
      KOKKOS_LAMBDA(nnz_lno_t tid)
      {
        std::cout << "Hello from thread " << tid << '\n';
        nnz_lno_t* activeQ = q1.data();
        nnz_lno_t* nextQ = q2.data();
        nnz_lno_t* scratch = teamScratch + tid * maxDeg;
        if(tid == 0)
        {
          activeQ[0] = start;
          activeQSize() = 1;
          nextQSize() = 0;
        }
        //do until every node has been visited and labeled
        while(visitCounter() < numRows)
        {
          mem.team_barrier();
          if(tid == 0)
          {
            std::cout << ">> At top of main loop, queue: ";
            for(int i = 0; i < activeQSize(); i++)
            {
              std::cout << activeQ[i] << ' ';
            }
            std::cout << '\n';
          }
          for(size_type workSet = 0; workSet < activeQSize(); workSet += nthreads)
          {
            nnz_lno_t neiCount = 0;
            nnz_lno_t process = LNO_MAX;
            if(workSet + tid < activeQSize())
            {
              //get pointer to thread-local scratch space, which has size maxDeg
              //the node to process
              process = activeQ[workSet + tid];
              for(int i = 0; i < nthreads; i++)
              {
                if(tid == i)
                  std::cout << "Hello from thread " << tid << ", processing " << process << '\n';
                mem.team_barrier();
              }
              offset_t rowStart = rowmap(process);
              offset_t rowEnd = rowmap(process + 1);
              //build a list of all non-visited neighbors
              for(offset_t j = rowStart; j < rowEnd; j++)
              {
                nnz_lno_t col = colinds(j);
                if(visit(col) == NOT_VISITED && col != process)
                {
                  //std::cout << "  Hello from thread " << tid << ", recording neighbor " << col << '\n';
                  scratch[neiCount] = col;
                  neiCount++;
                }
              }
              /*
              std::cout << "Thread " << tid << " has " << neiCount << " neighbors to add to queue (pre-sorting): ";
              for(int i = 0; i < neiCount; i++)
                std::cout << scratch[i] << ' ';
              std::cout << '\n';
              */
              //insertion sort the neighbors in ascending order of degree
              for(nnz_lno_t j = 1; j < neiCount; j++)
              {
                //move scratch[j] left into the correct position
                nnz_lno_t jcol = scratch[j];
                offset_t jdeg = rowmap(jcol + 1) - rowmap(jcol);
                nnz_lno_t k = j - 1;
                for(; k > 0; k--)
                {
                  offset_t kdeg = rowmap(scratch[k] + 1) - rowmap(scratch[k]);
                  if(kdeg <= jdeg)
                  {
                    //jcol belongs at position k + 1
                    k++;
                    break;
                  }
                }
                for(nnz_lno_t shifting = j - 1; shifting >= k; shifting--)
                {
                  scratch[shifting + 1] = scratch[shifting];
                }
                scratch[k] = jcol;
              }
              /*
              std::cout << "Thread " << tid << " has " << neiCount << " neighbors to add to queue (post-sorting): ";
              for(int i = 0; i < neiCount; i++)
                std::cout << scratch[i] << ' ';
              std::cout << '\n';
              */
            }
            //fill updateThreads with LNO_MAX
            for(size_type i = tid * numRows / nthreads; i < (tid + 1) * numRows / nthreads; i++)
            {
              updateThreads(i) = LNO_MAX;
            }
            //must do at least one pass of tracking update threads
            Kokkos::single(Kokkos::PerTeam(mem),
            KOKKOS_LAMBDA()
            {
              updateDirty() = 1;
            });
            while(updateDirty())
            {
              mem.team_barrier();
              Kokkos::single(Kokkos::PerTeam(mem),
              KOKKOS_LAMBDA()
              {
                updateDirty() = 0;
              });
              mem.team_barrier();
              if(workSet + tid < activeQSize())
              {
                //go through rows in thread-local scratch
                //replace updateThreads(row) with tid, if tid is lower
                for(size_type i = 0; i < neiCount; i++)
                {
                  nnz_lno_t prevThread = Kokkos::safe_load(&updateThreads(scratch[i]));
                  if(prevThread > tid)
                  {
                    //if some other thread hasn't already changed the updateThreads element, replace it with tid
                    //if the cmp-xchg fails, some other thread is also updating that entry so more updates will be required.
                    //but ideally it will succeed most of the time
                    if(!Kokkos::atomic_compare_exchange_strong<nnz_lno_t>(&updateThreads(scratch[i]), prevThread, tid))
                    {
                      std::cout << "Need another trip through determining updateThreads (shouldn't happen in serial!)\n";
                      Kokkos::atomic_increment(&updateDirty());
                    }
                  }
                }
              }
            }
            mem.team_barrier();
            //each thread counts the number of entries it will actually add to nextQ
            size_type numToQueue = 0;
            for(size_type i = 0; i < neiCount; i++)
            {
              if(updateThreads(scratch[i]) == tid)
              {
                numToQueue++;
              }
            }
            updateThreadCounts(tid) = numToQueue;
            for(int i = 0; i < nthreads; i++)
            {
              if(tid == i)
              {
                std::cout << "Thread " << tid << " has " << numToQueue << " neighbors to add to queue: ";
                for(int i = 0; i < neiCount; i++)
                  std::cout << scratch[i] << ' ';
                std::cout << '\n';
              }
              mem.team_barrier();
            }
            mem.team_barrier();
            size_type queueUpdateOffset = 0;
            for(size_type i = 0; i < tid; i++)
            {
              queueUpdateOffset += updateThreadCounts(i);
            }
            //write out all queue updates in parallel (only for threads that worked this iteration)
            if(workSet + tid < activeQSize())
            {
              std::cout << "Thread " << tid << " is updating queue.\n";
              size_type nextQueueIter = 0;
              for(size_type i = 0; i < neiCount; i++)
              {
                nnz_lno_t toQueue = scratch[i];
                if(updateThreads(toQueue) == tid)
                {
                  std::cout << "  Adding " << toQueue << " at index " << nextQSize() + queueUpdateOffset + nextQueueIter << '\n';
                  visit(toQueue) = QUEUED;
                  nextQ[nextQSize() + queueUpdateOffset + nextQueueIter] = toQueue;
                  nextQueueIter++;
                }
              }
              //apply correct label to process
              std::cout << "Thread " << tid << " is labeling " << process << " with " << visitCounter() + tid << '\n';
              visit(process) = visitCounter() + tid;
            }
            mem.team_barrier();
            Kokkos::single(Kokkos::PerTeam(mem),
            KOKKOS_LAMBDA()
            {
              std::cout << "One thread is updating visit counter from " << visitCounter() << " to ";
              if(workSet + nthreads > activeQSize())
                visitCounter() += activeQSize() - workSet;
              else
                visitCounter() += nthreads;
              std::cout << visitCounter() << '\n';
              //update queue size by the number of nodes added this iteration
              size_type totalQueueAdded = 0;
              for(size_type i = 0; i < nthreads; i++)
              {
                totalQueueAdded += updateThreadCounts(i);
              }
              nextQSize() += totalQueueAdded;
            });
            mem.team_barrier();
          }
          if(tid == 0)
            std::cout << "Done processing queue, flipping queue buffers.\n";
          //(thread-local) swap queue buffers and tail pointers
          {
            auto temp = activeQ;
            activeQ = nextQ;
            nextQ = temp;
          }
          mem.team_barrier();
          Kokkos::single(Kokkos::PerTeam(mem),
          KOKKOS_LAMBDA()
          {
            std::cout << "Queue for next iteration has " << nextQSize() << " nodes.\n";
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
                  activeQ[0] = search;
                  activeQSize() = 1;
                  visit(search) = QUEUED;
                  std::cout << "WARNING: graph not connected. Resuming BFS from node " << search << '\n';
                  break;
                }
              }
            }
          });
          if(tid == 0)
            std::cout << ">> At bottom of main loop.\n\n\n";
        }
      });
    });
    return visit;
  }

  //Find a peripheral node, either
  nnz_lno_t find_peripheral(PeripheralMode mode)
  {
    if(mode == MIN_DEGREE)
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
    return 0;
  }

  perm_view_t rcm()
  {
    //find a peripheral node
    //TODO: make mode configurable, but keep MIN_DEGREE the default
    nnz_lno_t periph = find_peripheral(MIN_DEGREE);
    //run Cuthill-McKee BFS from periph
    perm_view_t visit = parallel_rcm_bfs(periph);
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


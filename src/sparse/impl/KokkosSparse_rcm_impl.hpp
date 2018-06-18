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

/*
template <typename HandleType, typename lno_row_view_t_, typename lno_nnz_view_t_>
typename lno_nnz_view_t::non_const_type RCM(
    typename lno_row_view_t_::size_type numRows, typename lno_row_view_t_::size_type numCols,
    lno_row_view_t rowmap, lno_nnz_view_t colinds)
 */

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

  typedef typename lno_nnz_view_t::non_const_type non_const_lno_nnz_view_t;

  typedef typename HandleType::row_lno_temp_work_view_t row_lno_temp_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_view_t row_lno_persistent_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_host_view_t row_lno_persistent_work_host_view_t; //Host view type

  typedef typename HandleType::nnz_lno_temp_work_view_t nnz_lno_temp_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_host_view_t nnz_lno_persistent_work_host_view_t; //Host view type

  typedef Kokkos::RangePolicy<MyExecSpace> my_exec_space;

  typedef Kokkos::RangePolicy<MyExecSpace> range_policy_t ;
  typedef Kokkos::TeamPolicy<MyExecSpace> team_policy_t ;
  typedef typename team_policy_t::member_type team_member_t ;

  typedef Kokkos::MinLoc<nnz_lno_t, nnz_lno_t, MyTempMemorySpace> MinLocReducer;
  typedef Kokkos::MaxLoc<nnz_lno_t, nnz_lno_t, MyTempMemorySpace> MaxLocReducer;
  typedef Kokkos::ValLocScalar<nnz_lno_t, nnz_lno_t> ValLoc;

  typedef nnz_lno_t LO;

  RCM(HandleType* handle_, size_type numRows_, size_type numCols_, lno_row_view_t rowmap_, lno_nnz_view_t colinds_)
    : handle(handle_), numRows(numRows_), numCols(numCols_),
      rowmap(rowmap_), colinds(colinds_)
  {}

  lno_row_view_t rowmap;
  lno_nnz_view_t colinds;
  size_type numRows;
  size_type numCols;
  HandleType* handle;

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
    non_const_lno_nnz_view_t visit("BFS visited nodes", numRows);
    //visitCounter atomically counts timestamps
    Kokkos::View<nnz_lno_t, MyTempMemorySpace> visitCounter("BFS visit counter (atomic)");
    //the visit queue
    //will process the elements in [qStart, qEnd) in parallel during each sweep
    //the queue doesn't need to be circular since each node is visited exactly once
    non_const_lno_nnz_view_t q("BFS queue", numRows);
    Kokkos::View<nnz_lno_t, MyTempMemorySpace> qStart("BFS frontier start index (last in queue)");
    Kokkos::View<nnz_lno_t, MyTempMemorySpace> qEnd("BFS frontier end index (next in queue)");
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
    size_type maxDeg = 0;
    Kokkos::parallel_reduce(range_policy_t(0, numRows),
      KOKKOS_LAMBDA(size_type i, size_type& lmaxDeg)
      {
        size_type nnz = rowmap(i + 1) - rowmap(i);
        if(nnz > lmaxDeg)
        {
          lmaxDeg = nnz;
        }
      }, maxDeg);
    return maxDeg;
  }

  //breadth-first search, producing a Cuthill-McKee ordering
  //nodes are labeled in the exact order they would be in a serial BFS,
  //and neighbors are visited in ascending order of degree
  non_const_lno_nnz_view_t parallel_rcm_bfs(nnz_lno_t start)
  {
    using std::cout;
    //need to know maximum degree to allocate scratch space for threads
    auto maxDeg = find_max_degree();
    const nnz_lno_t NOT_VISITED = 0;
    const nnz_lno_t QUEUED = 1;
    const nnz_lno_t START = 2;
    //view for storing the visit timestamps
    non_const_lno_nnz_view_t visit("BFS visited nodes", numRows);
    //visitCounter atomically counts timestamps
    Kokkos::View<nnz_lno_t, MyTempMemorySpace> visitCounter("BFS visit counter (atomic)");
    //the visit queue
    //process the elements in [qStart, qEnd) in parallel during each sweep
    //the queue doesn't need to be circular since each node is visited exactly once
    non_const_lno_nnz_view_t q("BFS queue", numRows);
    Kokkos::View<nnz_lno_t, MyTempMemorySpace> qStart("BFS frontier start index (last in queue)");
    Kokkos::View<nnz_lno_t, MyTempMemorySpace> qEnd("BFS frontier end index (next in queue)");
    //make a temporary policy just to figure out how many threads AUTO makes
    team_policy_t tempPolicy(1, Kokkos::AUTO());
    Kokkos::parallel_for(team_policy_t(1, Kokkos::AUTO()).set_scratch_size(0, Kokkos::PerTeam(tempPolicy.team_size() * maxDeg * sizeof(nnz_lno_t))),
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
        nnz_lno_t* teamScratch = (nnz_lno_t*) mem.team_shmem().get_shmem(mem.team_size() * maxDeg * sizeof(nnz_lno_t));
        nnz_lno_t* scratch = &teamScratch[tid * maxDeg];
        //all threads work until every node has been labeled
        cout << "Entering main loop of RCM BFS\n";
        while(qStart() != qEnd())
        {
          //loop over the frontier, giving each thread one node to process
          size_type workStart = qStart();
          size_type workEnd = qEnd();
          cout << "Current queue contents: ";
          for(int i = workStart; i < workEnd; i++)
            cout << q(i) << ' ';
          cout << '\n';
          mem.team_barrier();
          qStart() = qEnd();
          //inside this loop, qEnd will be advanced as neighbors are enqueued for next iteration
          for(size_type teamIndex = workStart; teamIndex < workEnd; teamIndex += mem.team_size())
          {
            if(teamIndex + tid < workEnd)
            {
              //first, find the list of non-visited neighbors and store in scratch
              size_type i = teamIndex + tid;
              //the node to process
              nnz_lno_t process = q(i);
              cout << "Processing node " << process << '\n';
              size_type rowStart = rowmap(process);
              size_type rowEnd = rowmap(process + 1);
              //build a list of all non-visited neighbors
              size_type neiCount = 0;
              for(size_type j = rowStart; j < rowEnd; j++)
              {
                nnz_lno_t col = colinds(j);
                if(visit(col) == NOT_VISITED && col != process)
                {
                  scratch[neiCount++] = col;
                }
              }
              cout << "Non-visited neighbors (unsorted): ";
              for(size_type j = 0; j < neiCount; j++)
                cout << scratch[j] << ' ';
              cout << '\n';
              //insertion sort the neighbors in ascending order of degree
              for(size_type j = 1; j < neiCount; j++)
              {
                //move scratch[j] left into the correct position
                size_type jdeg = rowmap(scratch[j] + 1) - rowmap(scratch[j]);
                size_type k;
                for(k = j - 1; k >= 0; k--)
                {
                  size_type kdeg = rowmap(scratch[k] + 1) - rowmap(scratch[k]);
                  if(jdeg > kdeg)
                  {
                    //scratch[j] belongs at position k + 1, stop
                    k++;
                    break;
                  }
                }
                nnz_lno_t jcol = scratch[j];
                for(size_type shifting = j - 1; shifting >= k; shifting--)
                {
                  scratch[shifting + 1] = scratch[shifting];
                }
                scratch[k] = jcol;
              }
              cout << "Non-visited neighbors (sorted): ";
              for(size_type j = 0; j < neiCount; j++)
                cout << scratch[j] << ' ';
              cout << '\n';
              //mark the end of the active neighbor list, if it is not full
              if(neiCount < maxDeg)
                scratch[neiCount] = Kokkos::ArithTraits<nnz_lno_t>::max();
            }
            //thread 0 performs the actual queue and label operations in serial for all threads (so no atomics needed)
            //this should be a small amount of work compared to finding neighbors in order of degree
            mem.team_barrier();
            if(tid == 0)
            {
              for(size_type thread = 0; thread < mem.team_size(); thread++)
              {
                if(teamIndex + thread >= workEnd)
                  break;
                nnz_lno_t* threadScratch = &teamScratch[thread * maxDeg];
                for(size_type neiIndex = 0; neiIndex < maxDeg; neiIndex++)
                {
                  if(threadScratch[neiIndex] == Kokkos::ArithTraits<nnz_lno_t>::max())
                  {
                    //reached end of list for this thread
                    break;
                  }
                  nnz_lno_t nei = threadScratch[neiIndex];
                  if(visit(nei) == NOT_VISITED)
                  {
                    //enqueue nei
                    visit(nei) = QUEUED;
                    q(qEnd()++) = nei;
                  }
                }
                //assign final label to thread's current vertex
                std::cout << "New label of node " << q(teamIndex + thread) << ": " << visitCounter() << '\n';
                visit(q(teamIndex + thread)) = visitCounter()++;
              }
              //have to handle the case where graph is not connected
              if(qStart() == qEnd() && visitCounter() != numRows + 2)
              {
                //queue empty but not all vertices labeled
                //add the first NOT_VISITED node to the queue
                for(size_type search = 0; search < numRows; search++)
                {
                  if(visit(search) == NOT_VISITED)
                  {
                    q(qEnd()++) = search;
                    break;
                  }
                }
              }
            }
            mem.team_barrier();
          }
        }
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
          auto nnzInRow = rowmap(i + 1) - rowmap(i);
          if(nnzInRow < lminloc.val)
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
      typedef Kokkos::MaxLoc<nnz_lno_t, nnz_lno_t, MyTempMemorySpace> MaxLocT;
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

  non_const_lno_nnz_view_t rcm()
  {
    std::cout << "Finding RCM permutation.\n";
    non_const_lno_nnz_view_t perm("RCM permutation", numRows);
    //find a peripheral node
    //TODO: make mode configurable, but DOUBLE_BFS still the default
    nnz_lno_t periph = find_peripheral(DOUBLE_BFS);
    std::cout << "Peripheral (starting) node is " << periph << '\n';
    //run Cuthill-McKee BFS from periph
    auto visit = parallel_rcm_bfs(periph);
    std::cout << "Did BFS.\n";
    //reverse the visit order (for "reverse" C-M), and remove the bias.
    //removing bias just shifts range down to [0, numRows)
    Kokkos::parallel_for(range_policy_t(0, numRows / 2),
      KOKKOS_LAMBDA(size_type i)
      {
        nnz_lno_t temp = visit(numRows - 1 - i) - 2;
        visit(numRows - 1 - i) = visit(i) - 2;
        visit(i) = temp;
      });
    std::cout << "Done.\n";
    return visit;
  }
};

}}  //KokkosSparse::Impl

#endif


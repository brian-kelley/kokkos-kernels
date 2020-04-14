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
#include <Kokkos_Random.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_Parallel_Reduce.hpp>
#include "KokkosBlas1_fill.hpp"
#include "KokkosGraph_Distance1Color.hpp"
#include "KokkosKernels_Uniform_Initialized_MemoryPool.hpp"

#ifndef _KOKKOS_PARTITIONING_IMP_HPP
#define _KOKKOS_PARTITIONING_IMP_HPP

namespace KokkosSparse{

namespace Impl{

//Fill a view such that v(i) = i
//Does the same thing as std::iota(begin, end)
template<typename View, typename Ordinal>
struct IotaFunctor
{
  IotaFunctor(View& v_) : v(v_) {}
  KOKKOS_INLINE_FUNCTION void operator()(const Ordinal i) const
  {
    v(i) = i;
  }
  View v;
};

template<typename Ordinal, typename View, typename Rowmap>
struct FindMinDegreeInLevel
{
  using Reducer = Kokkos::MinLoc<Ordinal, Ordinal>;
  using Reduction = typename Reducer::value_type;

  FindMinDegreeInLevel(const Rowmap& rowmap_, const View& vertexList_)
    : rowmap(rowmap_), vertexList(vertexList_)
  {}

  KOKKOS_INLINE_FUNCTION void operator()(Ordinal i, Reduction& lmin) const
  {
    Ordinal v = vertexList(i);
    Ordinal thisDegree = rowmap(v + 1) - rowmap(v);
    if(thisDegree < lmin.val)
    {
      lmin.val = thisDegree;
      lmin.loc = v;
    }
  }

  Rowmap rowmap;
  View vertexList;
};


//Breadth-first search applications:
//  * single-source shortest path (unweighted only for now)
//  * building level sets
//  * computing RCM order
//  * growing clusters from roots

template <typename HandleType>
struct BreadthFirstSearch
{
  using ExecSpace = typename HandleType::HandleExecSpace;
  using MemSpace = typename HandleType::HandlePersistentMemorySpace;
  using Offset = typename HandleType::size_type;
  using Ordinal = typedef typename HandleType::nnz_lno_t;
  using OrdinalView = Kokkos::View<Ordinal*, MemSpace>;
  using OrdinalView2D = Kokkos::View<Ordinal**, Kokkos::LayoutLeft, MemSpace>;
  using Rowmap = typename HandleType::lno_row_view_t;
  using Colinds = typename HandleType::lno_nnz_view_t;

  BreadthFirstSearch(const Rowmap& rowmap_, const Colinds& colinds_)
    : rowmap(rowmap_), colinds(colinds_)
  {
    if(rowmap.extent(0) == 0)
      nv = 0;
    else
      nv = rowmap.extent(0) - 1;
  }

  /*
  template<typename Scalar, typename WeightView>
  void sssp(Ordinal source, const OrdinalView& distances, const WeightView& weights)
  {
    //Pseudocode:
    //  Run core BFS algorithm from source, also building frontiers one at a time using scan
    //  Instead of including all unvisited neighbors in next frontier, just include all neighbors
    //    But when including a neighbor, only include it if self->neighbor is the last edge on the best known path from source to neighbor.
    //    Need to compute what that path cost would be and compare it to neighbor's current cost.
    //TODO!
  }
  */

  //Do breadth-first search from source, and store level sets as offsets and lists of vertices.
  //Return the number of levels.
  int levelSets(Ordinal source, const OrdinalView& levelOffsets, const OrdinalView& levelVerts)
  {
    //Pseudocode:
    //  levelOffsets(i+1) = levelOffsets(i) + len(worklist)
    //  append the current worklist to levelVerts (using a deep-copy from subview to subview)
    //  Run a BFS step to advance to frontier i+1, and produce nextWorklist
    //  worklist = nextWorklist
  }

  void rcm(const OrdinalView& ordering)
  {
    //Pseudocode:
    //  Find pseudoperipheral
    //  Construct level sets rooted at pseudoperipheral
    //  In parallel, sort each level set ascending by degree
    //  Then reverse the order
  }

  //The fundamental BFS functor: just builds worklist each iteration
  template<typename Ordinal, typename Offset, typename Rowmap, typename Colinds, typename OrdinalView, typename Worklist>
  struct BfsFunctor
  {
    KOKKOS_INLINE_FUNCTION void operator()(const Ordinal i, Ordinal& lcount, bool finalPass) const
    {
      //TODO: copy from level-set version when that is known to be correct
    }

    Ordinal nv;
    Rowmap rowmap;
    Colinds colinds;
    OrdinalView pred;  //Initially all nv, pred(i) marks the predecessor used to bring vert i into a frontier.
    Worklist worklist; //an nx2 array
    int w;             //selects the current column of worklist
  };

  //The BFS functor, with explicit level sets
  //Level set format is like CRS:
  //  if v = levelVerts(j) is in level L, levelOffsets(L) <= j < levelOffsets(L+1)
  //The root is the only vertex in level 0.
  template<typename Ordinal, typename Offset, typename Rowmap, typename Colinds, typename OrdinalView, typename Worklist>
  struct BfsLevelSetFunctor
  {
    KOKKOS_INLINE_FUNCTION void operator()(const Ordinal i, Ordinal& lcount, bool finalPass) const
    {
      Ordinal v = worklist(i, w);
      Offset rowBegin = rowmap(v);
      Offset rowEnd = rowmap(v + 1);
      for(Offset j = rowBegin; j < rowEnd; j++)
      {
        Ordinal nei = colinds(j);
        if(v == nei)
          continue;
        bool isPred = false;
        if(pred(nei) == v)
        {
          lcount++;
          isPred = true;
        }
        else if(pred(nei) == nv)
        {
          //This neighbor has not been assigned a predecessor yet,
          //so attempt to use this vertex
          if(nv == Kokkos::atomic_compare_exchange(&pred(nei), nv, v))
          {
            //cmp-xchg uniquely succeeded for this vertex, so v is predecssor to nei
            lcount++;
            isPred = true;
          }
        }
        if(finalPass && isPred)
        {
          //v is the unique predecessor of nei, and v is in the current frontier.
          //Therefore nei is in the next frontier, and lcount gives the index to insert it.
          worklist(lcount, 1 - w) = nei;
          levelVerts(
        }
      }
      if(finalPass && lcount == nv)
      {
        //unique case: 
      }
    }

    Ordinal nv;
    Ordinal level;
    Ordinal levelSetOffset; //This value is updated on host after each level set is constructed.
    OrdinalView levelOffsets;
    OrdinalView levelVerts;
    Rowmap rowmap;
    Colinds colinds;
    OrdinalView pred;  //Initially all nv, pred(i) marks the predecessor used to bring vert i into a frontier.
    Worklist worklist; //an nx2 array
    int w;             //selects the current column of worklist
  };

  void expandClusters(const OrdinalView& vertClusters, const OrdinalView& distances, const OrdinalView& rootList)
  {
    OrdinalView2D worklist(Kokkos::ViewAllocateWithoutInitializing("Cluster BFS Worklist"), nv, 2);
    //Initial worklist length is the root list
    Ordinal worklistLen = rootList.extent(0);
    Kokkos::deep_copy(Kokkos::subview(worklist, std::make_pair((Ordinal) 0, worklistLen), 0),
        rootList);
    //Create functor that 
    asdf funct(...);
    //Until worklist is empty...
    while(worklistLen)
    {
      //Run a scan that populates the next worklist with the next frontier
      //The final scan result is the length
      Kokkos::parallel_scan(range_policy_t(0, worklistLen), funct, worklistLen);
      //Flip worklist buffers
      funct.w = 1 - funct.w;
    }
  }

  nnz_lno_t pseudoperipheral(nnz_lno_t start)
  {
    OrdinalView levelOffsets("Level set offsets", nv);
    OrdinalView levelVerts("Level set vertices", nv);
    auto last2Offsets = Kokkos::subview(levelOffsets, std::make_pair(nlevels - 1, nlevels));
    auto last2OffsetsHost = Kokkos::create_mirror_view(last2Offsets);
    //Number of repetitions of BFS: Zoltan also does 2.
    //Other algorithms like GPS and George-Liu do MANY more level set constructions
    //than 2, but 2 is almost certainly good enough for here
    const int numBFS = 2;
    for(int iter; iter < numBFS; iter++)
    {
      //compute rooted level structure from start
      int nlevels = levelSets(start, levelOffsets, levelVerts);
      Kokkos::deep_copy(last2OffsetsHost, last2Offsets);
      //get the range of the last level
      typename Kokkos::MinLoc<Ordinal, Ordinal>::value_type minLocReduction;
      Kokkos::parallel_reduce(
          range_policy_t(last2OffsetsHost(0), last2OffsetsHost(1)),
          FindMinDegreeInLevel<Ordinal, OrdinalView, Rowmap>(rowmap, levelVerts), minLocReduction);
      //the "location" stored in the reduction is just a vertex, not an index into levelVerts
      start = minLocReduction.loc;
    }
    return start;
  }

private:

  Rowmap rowmap;
  Colinds colinds;
  Ordinal nv;
};

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

  typedef typename lno_nnz_view_t::const_type const_lno_nnz_view_t;
  typedef typename lno_nnz_view_t::non_const_type non_const_lno_nnz_view_t;

  typedef typename HandleType::row_lno_temp_work_view_t row_lno_temp_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_view_t row_lno_persistent_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_host_view_t row_lno_persistent_work_host_view_t; //Host view type

  typedef typename HandleType::nnz_lno_temp_work_view_t nnz_lno_temp_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_host_view_t nnz_lno_persistent_work_host_view_t; //Host view type

  typedef nnz_lno_persistent_work_view_t nnz_view_t;
  typedef Kokkos::View<nnz_lno_t, MyTempMemorySpace, Kokkos::MemoryTraits<0>> single_view_t;
  typedef Kokkos::View<nnz_lno_t, Kokkos::HostSpace, Kokkos::MemoryTraits<0>> single_view_host_t;

  typedef Kokkos::RangePolicy<MyExecSpace> my_exec_space;

  typedef Kokkos::RangePolicy<MyExecSpace> range_policy_t ;
  typedef Kokkos::TeamPolicy<MyExecSpace> team_policy_t ;
  typedef typename team_policy_t::member_type team_member_t ;

  typedef nnz_lno_t LO;

  RCM(size_type numRows_, lno_row_view_t& rowmap_, lno_nnz_view_t& colinds_)
    : numRows(numRows_), rowmap(rowmap_), colinds(colinds_)
  {}

  nnz_lno_t numRows;
  const_lno_row_view_t rowmap;
  const_lno_nnz_view_t colinds;

  template<typename Rowmap>
  struct MaxDegreeFunctor
  {
    typedef typename std::remove_cv<typename Rowmap::value_type>::type size_type;
    MaxDegreeFunctor(Rowmap& rowmap_) : r(rowmap_) {}
    KOKKOS_INLINE_FUNCTION void operator()(const size_type i, size_type& lmax) const
    {
      size_type ideg = r(i + 1) - r(i);
      if(ideg > lmax)
        lmax = ideg;
    }
    Rowmap r;
  };

  //simple parallel reduction to find max degree in graph
  size_type find_max_degree()
  {
    size_type maxDeg = 0;
    Kokkos::parallel_reduce(range_policy_t(0, numRows), MaxDegreeFunctor<const_lno_row_view_t>(rowmap), Kokkos::Max<size_type>(maxDeg));
    //max degree should be computed as an offset_t,
    //but must fit in a nnz_lno_t
    return maxDeg;
  }

  //Functor that does breadth-first search on a sparse graph.
  struct BfsFunctor
  {
    BfsFunctor(const WorkView& workQueue_, const WorkView& scratch_, const nnz_view_t& visit_, const const_lno_row_view_t& rowmap_, const const_lno_nnz_view_t& colinds_, const single_view_t& numLevels_, const nnz_view_t& threadNeighborCounts_, nnz_lno_t start_, nnz_lno_t numRows_)
      : workQueue(workQueue_), scratch(scratch_), visit(visit_), rowmap(rowmap_), colinds(colinds_), numLevels(numLevels_), threadNeighborCounts(threadNeighborCounts_), start(start_), numRows(numRows_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const team_member_t mem) const
    {
      const nnz_lno_t LNO_MAX = Kokkos::ArithTraits<nnz_lno_t>::max();
      const nnz_lno_t NOT_VISITED = LNO_MAX;
      const nnz_lno_t QUEUED = NOT_VISITED - 1;
      int nthreads = mem.team_size();
      nnz_lno_t tid = mem.team_rank();
      auto neighborList = Kokkos::subview(scratch, tid, Kokkos::ALL());
      //active and next indicate which buffer in workQueue holds the nodes in current/next frontiers, respectively
      //active, next and visitCounter are thread-local, but always kept consistent across threads
      int active = 0;
      int next = 1;
      nnz_lno_t visitCounter = 0;
      Kokkos::single(Kokkos::PerTeam(mem),
      [=]()
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
        nnz_lno_t workStart = tid * activeQSize / nthreads;
        nnz_lno_t workEnd = (tid + 1) * activeQSize / nthreads;
        //the maximum work batch size (among all threads)
        //the following loop contains barriers so all threads must iterate same # of times
        nnz_lno_t maxBatch = (activeQSize + nthreads - 1) / nthreads;
        for(nnz_lno_t loop = 0; loop < maxBatch; loop++)
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
              if(col < numRows && Kokkos::atomic_compare_exchange_strong<nnz_lno_t>(&visit(col), NOT_VISITED, QUEUED))
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
          for(nnz_lno_t i = 0; i < tid; i++)
          {
            queueUpdateOffset += threadNeighborCounts(i);
          }
          //write out all updates to next queue in parallel
          if(busy)
          {
            nnz_lno_t nextQueueIter = 0;
            for(nnz_lno_t i = 0; i < neiCount; i++)
            {
              nnz_lno_t toQueue = neighborList(i);
              visit(toQueue) = QUEUED;
              workQueue(next, nextQSize + queueUpdateOffset + nextQueueIter) = toQueue;
              nextQueueIter++;
            }
            //assign level to to process
            visit(process) = level;
          }
          nnz_lno_t totalAdded = 0;
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
          [=]()
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
      [=]
      {
        numLevels() = level - 1;
      });
    }

    WorkView workQueue;
    WorkView scratch;
    nnz_view_t visit;
    const_lno_row_view_t rowmap;
    const_lno_nnz_view_t colinds;
    single_view_t numLevels;
    nnz_view_t threadNeighborCounts;
    nnz_lno_t start;
    nnz_lno_t numRows;
  };

  //Parallel breadth-first search, producing level structure in (xadj, adj) form:
  //xadj(level) gives index in adj where level begins.
  //Returns the total number of levels, and sets xadj, adj and maxDeg.
  nnz_lno_t parallel_bfs(nnz_lno_t start, nnz_view_t& xadj, nnz_view_t& adj, nnz_lno_t& maxDeg, nnz_lno_t nthreads)
  {
    //need to know maximum degree to allocate scratch space for threads
    maxDeg = find_max_degree();
    //view for storing the visit timestamps
    nnz_view_t visit("BFS visited nodes", numRows);
    const nnz_lno_t LNO_MAX = Kokkos::ArithTraits<nnz_lno_t>::max();
    const nnz_lno_t NOT_VISITED = LNO_MAX;
    KokkosBlas::fill(visit, NOT_VISITED);
    //the visit queue
    //one of q1,q2 is active at a time and holds the nodes to process in next BFS level
    //elements which are LNO_MAX are just placeholders (nothing to process)
    Kokkos::View<nnz_lno_t**, MyTempMemorySpace, Kokkos::MemoryTraits<0>> workQueue("BFS queue (double buffered)", 2, numRows);
    nnz_view_t threadNeighborCounts("Number of nodes to queue on each thread", nthreads);
    single_view_t numLevels("# of BFS levels");
    single_view_host_t numLevelsHost("# of BFS levels");
    Kokkos::View<nnz_lno_t**, MyTempMemorySpace, Kokkos::MemoryTraits<0u>> scratch("Scratch buffer shared by threads", nthreads, maxDeg);
    Kokkos::parallel_for(team_policy_t(1, nthreads), BfsFunctor(workQueue, scratch, visit, rowmap, colinds, numLevels, threadNeighborCounts, start, numRows));
    Kokkos::deep_copy(numLevelsHost, numLevels);
    //now that level structure has been computed, construct xadj/adj
    KokkosKernels::Impl::create_reverse_map<nnz_view_t, nnz_view_t, MyExecSpace>
      (numRows, numLevelsHost(), visit, xadj, adj);
    return numLevelsHost();
  }

  struct CuthillMcKeeFunctor
  {
    typedef Kokkos::View<offset_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0u>> ScoreView;

    CuthillMcKeeFunctor(nnz_lno_t numLevels_, nnz_lno_t maxDegree_, const const_lno_row_view_t& rowmap_, const const_lno_nnz_view_t& colinds_, const ScoreView& scores_, const ScoreView& scoresAux_, const nnz_view_t& visit_, const nnz_view_t& xadj_, const nnz_view_t& adj_, const nnz_view_t& adjAux_)
      : numLevels(numLevels_), maxDegree(maxDegree_), rowmap(rowmap_), colinds(colinds_), scores(scores_), scoresAux(scoresAux_), visit(visit_), xadj(xadj_), adj(adj_), adjAux(adjAux_)
    {
      numRows = rowmap.extent(0) - 1;
    }

    KOKKOS_INLINE_FUNCTION void operator()(const team_member_t mem) const
    {
      int tid = mem.team_rank();
      int nthreads = mem.team_size();
      const nnz_lno_t LNO_MAX = Kokkos::ArithTraits<nnz_lno_t>::max();
      nnz_lno_t visitCounter = 0;
      for(nnz_lno_t level = 0; level < numLevels; level++)
      {
        //iterate over vertices in this level and compute
        //min predecessors (minimum-labeled vertices from previous level)
        nnz_lno_t levelOffset = xadj(level);
        nnz_lno_t levelSize = xadj(level + 1) - levelOffset;
        //compute as offset_t to avoid overflow, but the upper bound on
        //the scores is approx. numRows * maxDegree, which should be representable
        nnz_lno_t workStart = tid * levelSize / nthreads;
        nnz_lno_t workEnd = (tid + 1) * levelSize / nthreads;
       for(nnz_lno_t i = workStart; i < workEnd; i++)
        {
          nnz_lno_t process = adj(levelOffset + i);
          nnz_lno_t minNeighbor = LNO_MAX;
          offset_t rowStart = rowmap(process);
          offset_t rowEnd = rowmap(process + 1);
          for(offset_t j = rowStart; j < rowEnd; j++)
          {
            nnz_lno_t neighbor = colinds(j);
            if(neighbor < numRows)
            {
              nnz_lno_t neighborVisit = visit(neighbor);
              if(neighborVisit < minNeighbor)
                minNeighbor = neighborVisit;
            }
          }
          scores(i) = ((offset_t) minNeighbor * (maxDegree + 1)) + (rowmap(process + 1) - rowmap(process));
        }
        mem.team_barrier();
        Kokkos::single(Kokkos::PerTeam(mem),
        [=]()
        {
          radixSortKeysAndValues<size_type, offset_t, nnz_lno_t, nnz_lno_t, team_member_t>
            (scores.data(), scoresAux.data(), adj.data() + levelOffset, adjAux.data(), levelSize, mem);
        });
        mem.team_barrier();
        //label all vertices (which are now in label order within their level)
        for(nnz_lno_t i = workStart; i < workEnd; i++)
        {
          nnz_lno_t process = adj(levelOffset + i);
          //visit counter increases with levels, so flip the range for the "reverse" in RCM
          visit(process) = visitCounter + i;
        }
        visitCounter += levelSize;
      }
    }

    nnz_lno_t numRows;
    nnz_lno_t numLevels;
    nnz_lno_t maxDegree;
    const_lno_row_view_t rowmap;
    const_lno_nnz_view_t colinds;
    ScoreView scores;
    ScoreView scoresAux;
    nnz_view_t visit;
    //The levels, stored in CRS format.
    //xadj stores offsets for each level, and adj stores the rows in each level.
    nnz_view_t xadj;
    nnz_view_t adj;
    nnz_view_t adjAux;
  };

  //Does the reversing in "reverse Cuthill-McKee")
  struct OrderReverseFunctor
  {
    OrderReverseFunctor(const nnz_view_t& visit_, nnz_lno_t numRows_)
      : visit(visit_), numRows(numRows_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
    {
      visit(i) = numRows - visit(i) - 1;
    }
    nnz_view_t visit;
    nnz_lno_t numRows;
  };

  //breadth-first search, producing a reverse Cuthill-McKee ordering
  nnz_view_t parallel_cuthill_mckee(nnz_lno_t start)
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
    Kokkos::parallel_reduce(range_policy_t(0, numLevels), MaxDegreeFunctor<nnz_view_t>(xadj), Kokkos::Max<nnz_lno_t>(maxLevelSize));
    //visit (to be returned) contains the RCM numberings of each row
    nnz_view_t visit("RCM labels", numRows);
    //Populate visit wth LNO_MAX so that the "min-labeled neighbor"
    //is always a node in the previous level
    const nnz_lno_t LNO_MAX = Kokkos::ArithTraits<nnz_lno_t>::max();
    KokkosBlas::fill(visit, LNO_MAX);
    //the "score" of a node is a single value that provides an ordering equivalent
    //to sorting by min predecessor and then by min degree
    //reduce nthreads to be a power of 2
    Kokkos::View<offset_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0u>> scores("RCM scores for sorting", maxLevelSize);
    Kokkos::View<offset_t*, MyTempMemorySpace, Kokkos::MemoryTraits<0u>> scoresAux("RCM scores for sorting (radix sort aux)", maxLevelSize);
    nnz_view_t adjAux("RCM scores for sorting (radix sort aux)", maxLevelSize);
    Kokkos::parallel_for(team_policy_t(1, nthreads), CuthillMcKeeFunctor(numLevels, maxDegree, rowmap, colinds, scores, scoresAux, visit, xadj, adj, adjAux));
    //reverse the visit order (for the 'R' in RCM)
    Kokkos::parallel_for(range_policy_t(0, numRows), OrderReverseFunctor(visit, numRows));
    return visit;
  }

  template<typename Reducer>
  struct MinDegreeRowFunctor
  {
    typedef typename Reducer::value_type Value;
    MinDegreeRowFunctor(const const_lno_row_view_t& rowmap_) : rowmap(rowmap_) {}
    KOKKOS_INLINE_FUNCTION void operator()(const size_type i, Value& lval) const
    {
      size_type ideg = rowmap(i + 1) - rowmap(i);
      if(ideg < lval.val)
      {
        lval.val = ideg;
        lval.loc = i;
      }
    }
    const_lno_row_view_t rowmap;
  };

  //parallel-for functor that assigns a cluster given a envelope-reduced reordering (like RCM)
  struct OrderToClusterFunctor
  {
    OrderToClusterFunctor(const nnz_view_t& ordering_, const nnz_view_t& vertClusters_, nnz_lno_t clusterSize_)
      : ordering(ordering_), vertClusters(vertClusters_), clusterSize(clusterSize_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
    {
      vertClusters(i) = ordering(i) / clusterSize;
    }

    const nnz_view_t ordering;
    nnz_view_t vertClusters;
    nnz_lno_t clusterSize;
  };

  //Find a peripheral node (one of minimal degree), suitable for starting RCM or BFS 
  nnz_lno_t find_peripheral()
  {
    typedef Kokkos::MinLoc<size_type, size_type> MinLocReducer;
    typedef typename MinLocReducer::value_type MinLocVal;
    MinLocVal v;
    Kokkos::parallel_reduce(range_policy_t(0, numRows),
        MinDegreeRowFunctor<MinLocReducer>(rowmap), MinLocReducer(v));
    return v.loc;
  }

  nnz_view_t cuthill_mckee()
  {
    nnz_lno_t periph = find_peripheral();
    //run Cuthill-McKee BFS from periph
    auto ordering = parallel_cuthill_mckee(periph);
    return ordering;
  }

  nnz_view_t rcm()
  {
    nnz_view_t cm = cuthill_mckee();
    //reverse the visit order (for the 'R' in RCM)
    Kokkos::parallel_for(range_policy_t(0, numRows), OrderReverseFunctor(cm, numRows));
    return cm;
  }

  nnz_view_t cm_cluster(nnz_lno_t clusterSize)
  {
    nnz_view_t cm = cuthill_mckee();
    nnz_view_t vertClusters("Vert to cluster", numRows);
    OrderToClusterFunctor makeClusters(cm, vertClusters, clusterSize);
    Kokkos::parallel_for(range_policy_t(0, numRows), makeClusters);
    return vertClusters;
  }
};

template <typename HandleType, typename lno_row_view_t, typename lno_nnz_view_t>
struct BFS
{
  typedef typename HandleType::HandleExecSpace ExecSpace;
  typedef typename HandleType::HandlePersistentMemorySpace MemorySpace;

  typedef typename HandleType::size_type size_type;
  typedef typename HandleType::nnz_lno_t nnz_lno_t;

  typedef typename lno_row_view_t::value_type offset_t;

  typedef typename HandleType::row_lno_temp_work_view_t row_lno_temp_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_view_t row_lno_persistent_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_host_view_t row_lno_persistent_work_host_view_t; //Host view type

  typedef typename HandleType::nnz_lno_temp_work_view_t nnz_lno_temp_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_host_view_t nnz_lno_persistent_work_host_view_t; //Host view type

  typedef nnz_lno_persistent_work_view_t nnz_view_t;
  typedef Kokkos::Bitset<MyExecSpace> bitset_t;

  typedef Kokkos::RangePolicy<MyExecSpace> range_policy_t;
  typedef Kokkos::TeamPolicy<MyExecSpace> team_policy_t;
  typedef typename team_policy_t::member_type team_member_t;

  // TBD: MemoryUnmanaged should be the default for shared memory space.
  using SharedStack = Kokkos::View<nnz_lno_t*, typename ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;

  //IMPORTANT: use this with vector length = 1 only, since each work item needs its own stack
  KOKKOS_INLINE_FUNCTION void operator()(team_member_t t) const
  {
    SharedStack vertStack(t.thread_scratch(0), maxDepth);
    SharedStack offsetStack(t.thread_scratch(0), maxDepth);
    nnz_lno_t workID = t.league_rank() * t.team_size() + t.team_rank();
    vertStack(0) = rootList(workID);
    offsetStack(0) = 0;
    //index of the stack top - always in [0, maxDepth)
    int top = 0;
    while(top >= 0)
    {
      //fetch the next neighbor of the stack top
      nnz_lno_t v = vertStack(top);
      if(rowmap(v + 1) - rowmap(v) == offsetStack(top))
      {
        //done with v, pop
        top--;
      }
      nnz_lno_t nei = colinds(rowmap(v) + offsetStack(top)++);
      if(nei == v)
        continue;
    }
  }

  size_t team_shmem_size(int teamSize) const
  {
    return 2 * maxDepth * teamSize;
  }

  int maxDepth;
};

template<typename HandleType, typename lno_row_view_t, typename lno_nnz_view_t>
void multiSourceDFS(
    const lno_row_view_t& rowptrs, const lno_nnz_view_t& colinds,
    const lno_nnz_view_t& vertClusters, const lno_nnz_view_t& rootList)
{

}

template <typename HandleType, typename lno_row_view_t, typename lno_nnz_view_t>
struct BalloonClustering
{
  typedef typename HandleType::HandleExecSpace MyExecSpace;
  typedef typename HandleType::HandleTempMemorySpace MyTempMemorySpace;
  typedef typename HandleType::HandlePersistentMemorySpace MyPersistentMemorySpace;

  typedef typename HandleType::size_type size_type;
  typedef typename HandleType::nnz_lno_t nnz_lno_t;

  typedef typename lno_row_view_t::value_type offset_t;

  typedef typename HandleType::row_lno_temp_work_view_t row_lno_temp_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_view_t row_lno_persistent_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_host_view_t row_lno_persistent_work_host_view_t; //Host view type

  typedef typename HandleType::nnz_lno_temp_work_view_t nnz_lno_temp_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_host_view_t nnz_lno_persistent_work_host_view_t; //Host view type

  typedef nnz_lno_persistent_work_view_t nnz_view_t;
  typedef Kokkos::View<float*, MyPersistentMemorySpace> float_view_t;

  typedef Kokkos::RangePolicy<MyExecSpace> my_exec_space;
  typedef Kokkos::Bitset<MyExecSpace> bitset_t;

  typedef Kokkos::RangePolicy<MyExecSpace> range_policy_t;
  typedef Kokkos::TeamPolicy<MyExecSpace> team_policy_t;
  typedef typename team_policy_t::member_type team_member_t;

  BalloonClustering(size_type numRows_, const lno_row_view_t& rowmap_, const lno_nnz_view_t& colinds_)
    : numRows(numRows_), rowmap(rowmap_), colinds(colinds_), randPool(0xDEADBEEF)
  {}

  nnz_lno_t numRows;
  lno_row_view_t rowmap;
  lno_nnz_view_t colinds;

  typedef Kokkos::Random_XorShift64_Pool<MyExecSpace> RandPool;
  RandPool randPool;

  struct InitRootsTag {};   //select roots; set their distances to 0
  struct InitialBFSTag {};  //do breadth-first search from roots. Only mark cluster verts if the distance-to-root improves.
  struct BuildBFSListTag {};//par-scan that builds the worklist for next iteration of BFS
  struct InitialDFSTag {};  //do depth-first search from roots, out to a fixed distance. Only mark cluster verts if the distance-to-root improves.
  struct RandomFillTag {};  //assign non-roots to random clusters, and assign large random distances
  struct UpdatePressureTag {};
  struct BalloonTag {};     //run the "balloon" procedure, where each cluster tries to inflate up to clusterSize

  struct BalloonFunctor 
  {
    BalloonFunctor(const nnz_view_t& vertClusters_, const nnz_view_t& clusterCounts_, const nnz_view_t& distances_, const lno_row_view_t& row_map_, const lno_nnz_view_t& col_inds_, const float_view_t& pressure_, nnz_lno_t clusterSize_, RandPool& randPool_)
      : vertClusters(vertClusters_), clusterCounts(clusterCounts_), distances(distances_), row_map(row_map_), col_inds(col_inds_), pressure(pressure_), clusterSize(clusterSize_), numRows(row_map.extent(0) - 1), vertLocks(numRows), randPool(randPool_)
    {
      numClusters = (numRows + clusterSize - 1) / clusterSize;
      avgClusterSize = (double) numRows / numClusters;
      iter = 0;
    }

    //Run init version over the number of clusters.
    KOKKOS_INLINE_FUNCTION void operator()(const InitRootsTag, const nnz_lno_t i) const
    {
      nnz_lno_t root;
      auto state = randPool.get_state();
      do
      {
        root = state.rand(numRows);
      }
      while(!Kokkos::atomic_compare_exchange_strong(&vertClusters(root), numClusters, i));
      randPool.free_state(state);
      distances(root) = 0;
      pressure(root) = 1;
    }

    KOKKOS_INLINE_FUNCTION void operator()(const InitialBFSTag, const nnz_lno_t i) const
    {
      nnz_lno_t root;
      auto state = randPool.get_state();
      do
      {
        root = state.rand(numRows);
      }
      while(!Kokkos::atomic_compare_exchange_strong(&vertClusters(root), numClusters, i));
      randPool.free_state(state);
      distances(root) = 0;
      pressure(root) = 1;
    }

    KOKKOS_INLINE_FUNCTION void operator()(const RandomFillTag, const nnz_lno_t i) const
    {
      if(vertClusters(i) == numClusters)
      {
        auto state = randPool.get_state();
        nnz_lno_t cluster = state.rand(numClusters);
        randPool.free_state(state);
        vertClusters(i) = cluster;
        Kokkos::atomic_increment(&clusterCounts(cluster));
        distances(i) = numRows;
        pressure(i) = 0.1;
      }
    };

    KOKKOS_INLINE_FUNCTION void operator()(const UpdatePressureTag, const nnz_lno_t i) const
    {
      nnz_lno_t cluster = vertClusters(i);
      if(cluster == numClusters)
      {
        //unassigned vertices have 0 pressure
        return;
      }
      //count the number of neighbors in the same cluster
      nnz_lno_t sameClusterNeighbors = 0;
      for(size_type j = row_map(i); j < row_map(i + 1); j++)
      {
        nnz_lno_t nei = col_inds(j);
        if(nei < numRows && nei != i && vertClusters(nei) == cluster)
        {
          //while we're at it, minimize distance to root as in Djikstra's
          if(distances(nei) + 1 < distances(i))
            distances(i) = distances(nei) + 1;
          sameClusterNeighbors++;
        }
      }
      nnz_lno_t curSize = clusterCounts(cluster);
      //update pressure, if cluster is undersized
      nnz_lno_t shortage = clusterSize - curSize;
      float pressureChange = (shortage * shortage * (1.0f + 0.2f * sameClusterNeighbors)) / (1.0f + distances(i));
      if(shortage > 0)
        pressure(i) += pressureChange;
    }

    KOKKOS_INLINE_FUNCTION void operator()(const BalloonTag, const nnz_lno_t i, double& sizeDeviation) const
    {
      nnz_lno_t cluster = vertClusters(i);
      if(cluster == numClusters)
        return;
      //find the weakest affinity neighbor
      nnz_lno_t weakNei = numRows;
      float weakestPressure = pressure(i);
      nnz_lno_t weakNeiCluster = numClusters;
      for(size_type j = row_map(i); j < row_map(i + 1); j++)
      {
        nnz_lno_t nei = col_inds(j);
        //to annex another vertex, it must be a non-root in a different cluster
        if(nei < numRows && nei != i && vertClusters(nei) != cluster && pressure(nei) < weakestPressure && distances(nei) != 0)
        {
          weakNei = nei;
          weakestPressure = pressure(nei);
          weakNeiCluster = vertClusters(nei);
        }
      }
      if(weakNei != numRows && clusterCounts(cluster) < clusterSize)
      {
        //this cluster will take over weakNei
        if(vertLocks.set(i))
        {
          if(vertLocks.set(weakNei))
          {
            Kokkos::atomic_increment(&clusterCounts(cluster));
            if(weakNeiCluster != numClusters)
              Kokkos::atomic_decrement(&clusterCounts(weakNeiCluster));
            vertClusters(weakNei) = cluster;
            pressure(i) -= pressure(weakNei);
            pressure(weakNei) = pressure(i);
            distances(weakNei) = distances(i) + 1;
            vertLocks.reset(weakNei);
          }
          vertLocks.reset(i);
        }
      }
      if(distances(i) == 0)
      {
        //roots update sizeDeviation on behalf of the cluster
        double deviation = clusterCounts(cluster) - avgClusterSize;
        sizeDeviation += deviation * deviation;
      }
    }

    nnz_view_t vertClusters;
    nnz_view_t clusterCounts;
    nnz_view_t distances;
    //row_map/col_inds of input graph (read-only)
    lno_row_view_t row_map;
    lno_nnz_view_t col_inds;
    float_view_t pressure;
    //constants
    nnz_lno_t clusterSize;
    nnz_lno_t numClusters;
    nnz_lno_t numRows;
    nnz_lno_t iter;
    Kokkos::Bitset<MyExecSpace> vertLocks;
    RandPool randPool;
    double avgClusterSize;
  };

  nnz_view_t run(nnz_lno_t clusterSize)
  {
    nnz_view_t vertClusters("Vertex cluster labels", numRows);
    //For the sake of completeness, handle the clusterSize = 1 case by generating a trivial (identity) clustering.
    if(clusterSize == 1)
    {
      Kokkos::parallel_for(Kokkos::RangePolicy<MyExecSpace>(0, numRows), IotaFunctor<nnz_view_t, nnz_lno_t>(vertClusters));
      return vertClusters;
    }
    nnz_lno_t numClusters = (numRows + clusterSize - 1) / clusterSize;
    nnz_view_t distances("Root distances", numRows);
    nnz_view_t clusterCounts("Vertices per cluster", numClusters);
    float_view_t pressure("Cluster pressure", numRows);
    Kokkos::deep_copy(clusterCounts, 1);
    Kokkos::deep_copy(vertClusters, (nnz_lno_t) numClusters);
    Kokkos::deep_copy(distances, numRows);
    BalloonFunctor funct(vertClusters, clusterCounts, distances, rowmap, colinds, pressure, clusterSize, randPool);
    Kokkos::Impl::Timer globalTimer;
    Kokkos::Impl::Timer timer;
    timer.reset();
    Kokkos::parallel_for(Kokkos::RangePolicy<MyExecSpace, InitRootsTag>(0, numClusters), funct);
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
    MyExecSpace().fence();
    std::cout << "Creating roots: " << timer.seconds() << '\n';
    timer.reset();
#endif
    double stoppingRMS = sqrt(numClusters * (0.02 * clusterSize) * (0.02 * clusterSize));
    double deviation = (double) numClusters * (clusterSize - 1) * (clusterSize - 1);
    int regressions = 0;
    while(true)
    {
      Kokkos::parallel_for(Kokkos::RangePolicy<MyExecSpace, UpdatePressureTag>(0, numRows), funct);
      double iterDeviation = 0;
      Kokkos::parallel_reduce(Kokkos::RangePolicy<MyExecSpace, BalloonTag>(0, numRows), funct, Kokkos::Sum<double>(iterDeviation));
      if(iterDeviation <= stoppingRMS || iterDeviation == deviation)
      {
        //got within 2% RMS of optimal, or stagnated
        deviation = iterDeviation;
        break;
      }
      else if(iterDeviation >= deviation)
      {
        regressions++;
        if(regressions == 3)
        {
          deviation = iterDeviation;
          break;
        }
      }
      deviation = iterDeviation;
      funct.iter++;
    }
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
    MyExecSpace().fence();
    std::cout << "Expanding clusters for " << funct.iter << " iterations: " << timer.seconds() << '\n';
    timer.reset();
#endif
    Kokkos::parallel_for(Kokkos::RangePolicy<MyExecSpace, RandomFillTag>(0, numRows), funct);
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
    MyExecSpace().fence();
    std::cout << "Randomly assigning clusters to remaining: " << timer.seconds() << '\n';
    std::cout << "Clustering total: " << globalTimer.seconds() << "\n\n";
#endif
    return vertClusters;
  }
};

}}  //KokkosSparse::Impl

#endif


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

#ifndef _KOKKOSCGSIMP_HPP
#define _KOKKOSCGSIMP_HPP

#include "KokkosKernels_Utils.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <Kokkos_Timer.hpp>
#include <Kokkos_Bitset.hpp>
#include <Kokkos_Sort.hpp>
#include "KokkosGraph_Distance1Color.hpp"
#include "KokkosKernels_BitUtils.hpp"
#include "KokkosKernels_SimpleUtils.hpp"
#include "KokkosSparse_partitioning_impl.hpp"
#include "KokkosGraph_MIS2.hpp"
#include "KokkosSparse_compressed_cluster_matrix.hpp"

namespace KokkosSparse
{
  namespace Impl
  {
    template <typename HandleType, typename in_rowmap_t, typename in_entries_t, typename in_values_t>
    class ClusterGaussSeidel
    {
    public:

      using CGSHandle       = typename HandleType::ClusterGaussSeidelHandleType;

      //Primitve types
      using KAT             = Kokkos::Details::ArithTraits<scalar_t>;
      using size_type       = typename CGSHandle::size_type;
      using lno_t           = typename CGSHandle::lno_t;
      using scalar_t        = typename CGSHandle::scalar_t;
      using mag_t           = typename KAT::mag_type;
      using unit_t          = typename CGSHandle::unit_t;
      using color_t         = typename HandleType::GraphColoringHandleType::color_t;

      //Views and containers
      using offset_view_t   = typename CGSHandle::offset_view_t;
      using ordinal_view_t  = typename CGSHandle::ordinal_view_t;
      using scalar_view_t   = typename CGSHandle::scalar_view_t;
      using mag_view_t      = Kokkos::View<mag_t*, mem_space>;
      using color_view_t    = typename HandleType::GraphColoringHandleType::color_view_t;
      using bitset_t        = Kokkos::Bitset<exec_space>;
      using const_bitset_t  = Kokkos::ConstBitset<exec_space>;

      //Device types
      using exec_space      = typename CGSHandle::exec_space;
      using mem_space       = typename CGSHandle::mem_space;
      using device_t        = Kokkos::Device<exec_space, mem_space>;
      using range_policy_t  = Kokkos::RangePolicy<exec_space>;
      using team_policy_t   = Kokkos::TeamPolicy<exec_space>;
      using team_member_t   = typename team_policy_t::member_type;

    private:
      HandleType *handle;

      //Get the specialized ClusterGaussSeidel handle from the main handle
      CGSHandle* get_gs_handle()
      {
        auto *gsHandle = dynamic_cast<CGSHandle*>(this->handle->get_gs_handle());
        if(!gsHandle)
        {
          throw std::runtime_error("ClusterGaussSeidel: GS handle has not been created, or is set up for Point GS.");
        }
        return gsHandle;
      }

      lno_t num_rows;
      lno_t num_cols;
      in_rowmap_t row_map;
      in_entries_t entries;
      in_values_t values;

    public:

      /**
       * \brief constructor
       */

      ClusterGaussSeidel(HandleType *handle_,
                  lno_t num_rows_,
                  lno_t num_cols_,
                  const_lno_row_view_t row_map_,
                  const_lno_nnz_view_t entries_,
                  const_scalar_nnz_view_t values_):
        handle(handle_), num_rows(num_rows_), num_cols(num_cols_),
        row_map(row_map_), entries(entries_), values(values_),
        have_diagonal_given(false),
        is_symmetric(true)
      {}

      ClusterGaussSeidel(HandleType *handle_,
                  lno_t num_rows_,
                  lno_t num_cols_,
                  const_lno_row_view_t row_map_,
                  const_lno_nnz_view_t entries_,
                  bool is_symmetric_ = true):
        handle(handle_),
        num_rows(num_rows_), num_cols(num_cols_),
        row_map(row_map_),
        entries(entries_),
        values(),
        have_diagonal_given(false),
        is_symmetric(is_symmetric_)
      {}

      /**
       * \brief constructor
       */
      ClusterGaussSeidel(HandleType *handle_,
                  lno_t num_rows_,
                  lno_t num_cols_,
                  const_lno_row_view_t row_map_,
                  const_lno_nnz_view_t entries_,
                  const_scalar_nnz_view_t values_,
                  bool is_symmetric_):
        handle(handle_),
        num_rows(num_rows_), num_cols(num_cols_),
        row_map(row_map_), entries(entries_), values(values_),
        have_diagonal_given(false),
        is_symmetric(is_symmetric_)
      {}

      ClusterGaussSeidel(
                  HandleType *handle_,
                  lno_t num_rows_,
                  lno_t num_cols_,
                  const_lno_row_view_t row_map_,
                  const_lno_nnz_view_t entries_,
                  const_scalar_nnz_view_t values_,
                  const_scalar_nnz_view_t given_inverse_diagonal_,
                  bool is_symmetric_):
        handle(handle_),
        num_rows(num_rows_), num_cols(num_cols_),
        row_map(row_map_), entries(entries_), values(values_),
        given_inverse_diagonal(given_inverse_diagonal_),
        have_diagonal_given(true),
        is_symmetric(is_symmetric_)
      {}

      //Functor to swap the numbers of two colors,
      //so that the last cluster has the last color.
      //Except, doesn't touch the color of the last cluster,
      //since that value is needed the entire time this is running.
      struct ClusterColorRelabelFunctor
      {
        typedef typename HandleType::GraphColoringHandleType GCHandle;
        typedef typename GCHandle::color_view_t ColorView;
        typedef Kokkos::View<row_lno_t*, MyTempMemorySpace> RowmapView;
        typedef Kokkos::View<lno_t*, MyTempMemorySpace> EntriesView;
        ClusterColorRelabelFunctor(ColorView& colors_, color_t numClusterColors_, lno_t numClusters_)
          : colors(colors_), numClusterColors(numClusterColors_), numClusters(numClusters_)
        {}

        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
        {
          if(colors(i) == numClusterColors)
            colors(i) = colors(numClusters - 1);
          else if(colors(i) == colors(numClusters - 1))
            colors(i) = numClusterColors;
        }

        ColorView colors;
        color_t numClusterColors;
        lno_t numClusters;
      };

      //Relabel the last cluster, after running ClusterColorRelabelFunctor.
      //Call with a one-element range policy.
      struct RelabelLastColorFunctor
      {
        typedef typename HandleType::GraphColoringHandleType GCHandle;
        typedef typename GCHandle::color_view_t ColorView;

        RelabelLastColorFunctor(ColorView& colors_, color_t numClusterColors_, lno_t numClusters_)
          : colors(colors_), numClusterColors(numClusterColors_), numClusters(numClusters_)
        {}

        KOKKOS_INLINE_FUNCTION void operator()(const size_type) const
        {
          colors(numClusters - 1) = numClusterColors;
        }
        
        ColorView colors;
        color_t numClusterColors;
        lno_t numClusters;
      };

      struct ClusterToVertexColoring
      {
        typedef typename HandleType::GraphColoringHandleType GCHandle;
        typedef typename GCHandle::color_view_t ColorView;

        ClusterToVertexColoring(ColorView& clusterColors_, ColorView& vertexColors_, lno_t numRows_, lno_t numClusters_, lno_t clusterSize_)
          : clusterColors(clusterColors_), vertexColors(vertexColors_), numRows(numRows_), numClusters(numClusters_), clusterSize(clusterSize_)
        {}

        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
        {
          size_type cluster = i / clusterSize;
          size_type clusterOffset = i - cluster * clusterSize;
          vertexColors(i) = ((clusterColors(cluster) - 1) * clusterSize) + clusterOffset + 1;
        }

        ColorView clusterColors;
        ColorView vertexColors;
        lno_t numRows;
        lno_t numClusters;
        lno_t clusterSize;
      };

      template<typename nnz_view_t>
      struct ClusterSizeFunctor
      {
        ClusterSizeFunctor(nnz_view_t& counts_, nnz_view_t& vertClusters_)
          : counts(counts_), vertClusters(vertClusters_)
        {}
        KOKKOS_INLINE_FUNCTION void operator()(const lno_t i) const
        {
          Kokkos::atomic_increment(&counts(vertClusters(i)));
        }
        nnz_view_t counts;
        nnz_view_t vertClusters;
      };

      template<typename nnz_view_t>
      struct FillClusterVertsFunctor
      {
        FillClusterVertsFunctor(nnz_view_t& clusterOffsets_, nnz_view_t& clusterVerts_, nnz_view_t& vertClusters_, nnz_view_t& insertCounts_)
          : clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), vertClusters(vertClusters_), insertCounts(insertCounts_)
        {}
        KOKKOS_INLINE_FUNCTION void operator()(const lno_t i) const
        {
          lno_t cluster = vertClusters(i);
          lno_t offset = clusterOffsets(cluster) + Kokkos::atomic_fetch_add(&insertCounts(cluster), 1);
          clusterVerts(offset) = i;
        }
        nnz_view_t clusterOffsets;
        nnz_view_t clusterVerts;
        nnz_view_t vertClusters;
        nnz_view_t insertCounts;
      };

      template<typename Rowmap, typename Colinds, typename nnz_view_t>
      struct BuildCrossClusterMaskFunctor
      {
        BuildCrossClusterMaskFunctor(Rowmap& rowmap_, Colinds& colinds_, nnz_view_t& clusterOffsets_, nnz_view_t& clusterVerts_, nnz_view_t& vertClusters_, bitset_t& mask_)
          : numRows(rowmap_.extent(0) - 1), rowmap(rowmap_), colinds(colinds_), clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), vertClusters(vertClusters_), mask(mask_)
        {}

        //Used a fixed-size hash set in shared memory
        KOKKOS_INLINE_FUNCTION constexpr int tableSize() const
        {
          //Should always be a power-of-two, so that X % tableSize() reduces to a bitwise and.
          return 512;
        }

        //Given a cluster index, get the hash table index.
        //This is the 32-bit xorshift RNG, but it works as a hash function.
        KOKKOS_INLINE_FUNCTION unsigned xorshiftHash(lno_t cluster) const
        {
          unsigned x = cluster;
          x ^= x << 13;
          x ^= x >> 17;
          x ^= x << 5;
          return x;
        }

        KOKKOS_INLINE_FUNCTION bool lookup(lno_t cluster, int* table) const
        {
          unsigned h = xorshiftHash(cluster);
          for(unsigned i = h; i < h + 2; i++)
          {
            if(table[i % tableSize()] == cluster)
              return true;
          }
          return false;
        }

        //Try to insert the edge between cluster (team's cluster) and neighbor (neighboring cluster)
        //by inserting nei into the table.
        KOKKOS_INLINE_FUNCTION bool insert(lno_t cluster, lno_t nei, int* table) const
        {
          unsigned h = xorshiftHash(nei);
          for(unsigned i = h; i < h + 2; i++)
          {
            if(Kokkos::atomic_compare_exchange_strong<int>(&table[i % tableSize()], cluster, nei))
              return true;
          }
          return false;
        }

        KOKKOS_INLINE_FUNCTION void operator()(const team_member_t t) const
        {
          lno_t cluster = t.league_rank();
          lno_t clusterSize = clusterOffsets(cluster + 1) - clusterOffsets(cluster);
          //Use a fixed-size hash table per thread to accumulate neighbor of the cluster.
          //If it fills up (very unlikely) then just count every remaining edge going to another cluster
          //not already in the table; this provides a reasonable upper bound for overallocating the cluster graph.
          //each thread handles a cluster
          int* table = (int*) t.team_shmem().get_shmem(tableSize() * sizeof(int));
          //mark every entry as cluster (self-loop) to represent free/empty
          Kokkos::parallel_for(Kokkos::TeamVectorRange(t, tableSize()),
            [&](const lno_t i)
            {
              table[i] = cluster;
            });
          t.team_barrier();
          //now, for each row belonging to the cluster, iterate through the neighbors
          Kokkos::parallel_for(Kokkos::TeamThreadRange(t, clusterSize),
            [&] (const lno_t i)
            {
              lno_t row = clusterVerts(clusterOffsets(cluster) + i);
              lno_t rowDeg = rowmap(row + 1) - rowmap(row);
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, rowDeg),
                [&] (const lno_t j)
                {
                  lno_t nei = colinds(rowmap(row) + j);
                  //Remote neighbors are not included
                  if(nei >= numRows)
                    return;
                  lno_t neiCluster = vertClusters(nei);
                  if(neiCluster != cluster)
                  {
                    //Have a neighbor. Try to find it in the table.
                    if(!lookup(neiCluster, table))
                    {
                      //Not in the table. Try to insert it.
                      insert(cluster, neiCluster, table);
                      //Whether or not insertion succeeded,
                      //this is a cross-cluster edge possibly not seen before
                      mask.set(rowmap(row) + j);
                    }
                  }
                });
            });
        }

        size_t team_shmem_size(int teamSize) const
        {
          return tableSize() * sizeof(int);
        }

        lno_t numRows;
        Rowmap rowmap;
        Colinds colinds;
        nnz_view_t clusterOffsets;
        nnz_view_t clusterVerts;
        nnz_view_t vertClusters;
        bitset_t mask;
      };

      template<typename Rowmap, typename Colinds, typename nnz_view_t>
      struct FillClusterEntriesFunctor
      {
        FillClusterEntriesFunctor(
            Rowmap& rowmap_, Colinds& colinds_, nnz_view_t& clusterRowmap_, nnz_view_t& clusterEntries_, nnz_view_t& clusterOffsets_, nnz_view_t& clusterVerts_, nnz_view_t& vertClusters_, bitset_t& edgeMask_)
          : rowmap(rowmap_), colinds(colinds_), clusterRowmap(clusterRowmap_), clusterEntries(clusterEntries_), clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), vertClusters(vertClusters_), edgeMask(edgeMask_)
        {}
        //Run this scan over entries in clusterVerts (reordered point rows)
        KOKKOS_INLINE_FUNCTION void operator()(const lno_t i, lno_t& lcount, const bool& finalPass) const
        {
          lno_t numRows = rowmap.extent(0) - 1;
          lno_t row = clusterVerts(i);
          size_type rowStart = rowmap(row);
          size_type rowEnd = rowmap(row + 1);
          lno_t cluster = vertClusters(row);
          lno_t clusterStart = clusterOffsets(cluster);
          //Count the number of entries in this row.
          //This is how much lcount will be increased by,
          //yielding the offset corresponding to
          //these point entries in the cluster entries.
          lno_t rowEntries = 0;
          for(size_type j = rowStart; j < rowEnd; j++)
          {
            if(edgeMask.test(j))
              rowEntries++;
          }
          if(finalPass)
          {
            //if this is the last row in the cluster, update the upper bound in clusterRowmap
            if(i == clusterStart)
            {
              clusterRowmap(cluster) = lcount;
            }
            lno_t clusterEdge = lcount;
            //populate clusterEntries for these edges
            for(size_type j = rowStart; j < rowEnd; j++)
            {
              if(edgeMask.test(j))
              {
                clusterEntries(clusterEdge++) = vertClusters(colinds(j));
              }
            }
          }
          //update the scan result at the end (exclusive)
          lcount += rowEntries;
          if(i == numRows - 1 && finalPass)
          {
            //on the very last row, set the last entry of the cluster rowmap
            clusterRowmap(clusterRowmap.extent(0) - 1) = lcount;
          }
        }
        Rowmap rowmap;
        Colinds colinds;
        nnz_view_t clusterRowmap;
        nnz_view_t clusterEntries;
        nnz_view_t clusterOffsets;
        nnz_view_t clusterVerts;
        nnz_view_t vertClusters;
        const_bitset_t edgeMask;
      };

      //Assign cluster labels to vertices, given that the vertices are naturally
      //ordered so that contiguous groups of vertices form decent clusters.
      template<typename View>
      struct NopVertClusteringFunctor
      {
        NopVertClusteringFunctor(View& vertClusters_, lno_t clusterSize_) :
            vertClusters(vertClusters_),
            numRows(vertClusters.extent(0)),
            clusterSize(clusterSize_)
        {}
        KOKKOS_INLINE_FUNCTION void operator()(const lno_t i) const
        {
          vertClusters(i) = i / clusterSize;
        }
        View vertClusters;
        lno_t numRows;
        lno_t clusterSize;
      };

      template<typename View>
      struct ReorderedClusteringFunctor
      {
        ReorderedClusteringFunctor(View& vertClusters_, View& ordering_, lno_t clusterSize_) :
            vertClusters(vertClusters_),
            ordering(ordering_),
            numRows(vertClusters.extent(0)),
            clusterSize(clusterSize_)
        {}
        KOKKOS_INLINE_FUNCTION void operator()(const lno_t i) const
        {
          vertClusters(i) = ordering(i) / clusterSize;
        }
        View vertClusters;
        View ordering;
        lno_t numRows;
        lno_t clusterSize;
      };


      void initialize_symbolic()
      {
        using raw_rowmap_t = Kokkos::View<const size_type*, mem_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using raw_colinds_t = Kokkos::View<const lno_t*, mem_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        CGSHandle* gsHandle = get_gs_handle();
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        Kokkos::Impl::Timer timer;
#endif
        //sym_xadj/sym_adj is only used here for clustering.
        //Create them as non-const, unmanaged views to avoid
        //duplicating a bunch of code between the
        //symmetric and non-symmetric input cases.
        offset_view_t sym_xadj;
        ordinal_view_t sym_adj;
        if(!this->is_symmetric)
        {
          KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap
            <in_rowmap_t, in_colinds_t, offset_view_t, ordinal_view_t, exec_space>
            (num_rows, this->row_map, this->entries, sym_xadj, sym_adj);
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
          std::cout << "SYMMETRIZING TIME: " << timer.seconds() << std::endl;
          timer.reset();
#endif
        }
        //Now that a symmetric graph is available, build the cluster graph (also symmetric)
        lno_t clusterSize = gsHandle->get_cluster_size();
        lno_t numClusters = (num_rows + clusterSize - 1) / clusterSize;
        ordinal_view_t clusterOffsets("Cluster offsets", numClusters + 1);
        ordinal_view_t clusterVerts("Cluster -> vertices", num_rows);
        raw_rowmap_t raw_sym_xadj;
        raw_colinds_t raw_sym_adj;
        if(this->is_symmetric)
        {
          raw_sym_xadj = raw_rowmap_t(this->row_map.data(), this->row_map.extent(0));
          raw_sym_adj = raw_colinds_t(this->entries.data(), this->entries.extent(0));
        }
        else
        {
          raw_sym_xadj = raw_rowmap_t(sym_xadj.data(), sym_xadj.extent(0));
          raw_sym_adj = raw_colinds_t(sym_adj.data(), sym_adj.extent(0));
        }
        ordinal_view_t vertClusters;
        auto clusterAlgo = gsHandle->get_clustering_algo();
        if(clusterAlgo == CLUSTER_DEFAULT)
          clusterAlgo = CLUSTER_MIS2;
        switch(clusterAlgo)
        {
          case CLUSTER_MIS2:
          {
            vertClusters = KokkosGraph::Experimental::graph_mis2_coarsen<MyExecSpace, raw_rowmap_t, raw_colinds_t, nnz_view_t>
              (raw_sym_xadj, raw_sym_adj, numClusters, KokkosGraph::MIS2_FAST);
            break;
          }
          case CLUSTER_BALLOON:
          {
            BalloonClustering<HandleType, raw_rowmap_t, raw_colinds_t> balloon(num_rows, raw_sym_xadj, raw_sym_adj);
            vertClusters = balloon.run(clusterSize);
            break;
          }
          case CLUSTER_DEFAULT:
          {
            throw std::logic_error("Logic to choose default clustering algorithm is incorrect");
          }
          default:
            throw std::runtime_error("Clustering algo " + std::to_string((int) clusterAlgo) + " is not implemented");
        }
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "Graph clustering: " << timer.seconds() << '\n';
        timer.reset();
#endif
        //Construct the cluster offset and vertex array. These allow fast iteration over all vertices in a given cluster.
        Kokkos::parallel_for(range_policy_t(0, num_rows), ClusterSizeFunctor<nnz_view_t>(clusterOffsets, vertClusters));
        KokkosKernels::Impl::exclusive_parallel_prefix_sum<nnz_view_t, MyExecSpace>(numClusters + 1, clusterOffsets);
        {
          nnz_view_t tempInsertCounts("Temporary cluster insert counts", numClusters);
          Kokkos::parallel_for(range_policy_t(0, num_rows), FillClusterVertsFunctor<nnz_view_t>(clusterOffsets, clusterVerts, vertClusters, tempInsertCounts));
        }
#if KOKKOSSPARSE_IMPL_PRINTDEBUG
        {
          auto clusterOffsetsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), clusterOffsets);
          auto clusterVertsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), clusterVerts);
          puts("Clusters (cluster #, and vertex #s):");
          for(lno_t i = 0; i < numClusters; i++)
          {
            printf("%d: ", (int) i);
            for(lno_t j = clusterOffsetsHost(i); j < clusterOffsetsHost(i + 1); j++)
            {
              printf("%d ", (int) clusterVerts(j));
            }
            putchar('\n');
          }
          printf("\n\n\n");
        }
#endif
        //Determine the set of edges (in the point graph) that cross between two distinct clusters
        int vectorSize = this->handle->get_suggested_vector_size(num_rows, raw_sym_adj.extent(0));
        bitset_t crossClusterEdgeMask(raw_sym_adj.extent(0));
        size_type numClusterEdges;
        {
          BuildCrossClusterMaskFunctor<raw_rowmap_t, raw_colinds_t, ordinal_view_t>
            buildEdgeMask(raw_sym_xadj, raw_sym_adj, clusterOffsets, clusterVerts, vertClusters, crossClusterEdgeMask);
          int sharedPerTeam = buildEdgeMask.team_shmem_size(0); //using team-size = 0 for since no per-thread shared is used.
          int teamSize = KokkosKernels::Impl::get_suggested_team_size<team_policy_t>(buildEdgeMask, vectorSize, sharedPerTeam, 0);
          Kokkos::parallel_for(team_policy_t(numClusters, teamSize, vectorSize).set_scratch_size(0, Kokkos::PerTeam(sharedPerTeam)), buildEdgeMask);
          numClusterEdges = crossClusterEdgeMask.count();
        }
        ordinal_view_t clusterRowmap = nnz_view_t("Cluster graph rowmap", numClusters + 1);
        ordinal_view_t clusterEntries = nnz_view_t("Cluster graph colinds", numClusterEdges);
        Kokkos::parallel_scan(range_policy_t(0, num_rows), FillClusterEntriesFunctor<raw_rowmap_t, raw_colinds_t, ordinal_view_t>
            (raw_sym_xadj, raw_sym_adj, clusterRowmap, clusterEntries, clusterOffsets, clusterVerts, vertClusters, crossClusterEdgeMask));
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "Building explicit cluster graph: " << timer.seconds() << '\n';
        timer.reset();
#endif
#if KOKKOSSPARSE_IMPL_PRINTDEBUG
        {
          auto clusterRowmapHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), clusterRowmap);
          auto clusterEntriesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), clusterEntries);
          puts("Cluster graph (cluster #, and neighbors):");
          for(lno_t i = 0; i < numClusters; i++)
          {
            printf("%d: ", (int) i);
            for(lno_t j = clusterRowmapHost(i); j < clusterRowmapHost(i + 1); j++)
            {
              printf("%d ", (int) clusterEntriesHost(j));
            }
            putchar('\n');
          }
          printf("\n\n\n");
        }
#endif
        //Get the coloring of the cluster graph.
        typename HandleType::GraphColoringHandleType::color_view_t colors;
        color_t numColors;
#if KOKKOSSPARSE_IMPL_RUNSEQUENTIAL
        numColors = numClusters;
        std::cout << "SEQUENTIAL CGS: numColors = numClusters = " << numClusters << '\n';
        typename HandleType::GraphColoringHandleType::color_view_t::HostMirror h_colors = Kokkos::create_mirror_view(colors);
        for(int i = 0; i < numClusters; ++i){
          h_colors(i) = i + 1;
        }
        Kokkos::deep_copy(colors, h_colors);
#else
        //Create a handle that uses lno_t as the size_type, since the cluster graph should never be larger than 2^31 entries.
        KokkosKernels::Experimental::KokkosKernelsHandle<lno_t, lno_t, double, exec_space, mem_space, mem_space> kh;
        kh.create_graph_coloring_handle(KokkosGraph::COLORING_DEFAULT);
        KokkosGraph::Experimental::graph_color_symbolic(&kh, numClusters, numClusters, clusterRowmap, clusterEntries);
        //retrieve colors
        auto coloringHandle = kh.get_graph_coloring_handle();
        colors = coloringHandle->get_vertex_colors();
        numColors = coloringHandle->get_num_colors();
        kh.destroy_graph_coloring_handle();
#endif
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "Coloring: " << timer.seconds() << '\n';
        timer.reset();
#endif
        ordinal_view_t color_xadj;
        ordinal_view_t color_adj;
        KokkosKernels::Impl::create_reverse_map
          <typename HandleType::GraphColoringHandleType::color_view_t,
           ordinal_view_t, exec_space>
          (numClusters, numColors, colors, color_xadj, color_adj);
        exec_space().fence();
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "CREATE_REVERSE_MAP:" << timer.seconds() << std::endl;
        timer.reset();
#endif
        nnz_lno_persistent_work_host_view_t color_xadj_host(Kokkos::ViewAllocateWithoutInitializing("Color xadj"), color_xadj.extent(0));
        Kokkos::deep_copy(color_xadj_host, color_xadj);
        gsHandle->set_color_xadj(color_xadj_host);
        gsHandle->set_color_adj(color_adj);
        gsHandle->set_num_colors(numColors);
        gsHandle->set_cluster_xadj(clusterOffsets);
        gsHandle->set_cluster_adj(clusterVerts);
        gsHandle->set_vert_clusters(vertClusters);
        gsHandle->set_call_symbolic(true);
      }

      //FlowOrderFunctor: greedily reorder vertices within a cluster to maximize
      //convergence in the forward direction.

      struct FlowOrderFunctor
      {
        FlowOrderFunctor(
            const ordinal_view_t& clusterOffsets_,
            const ordinal_view_t& clusterVerts_,
            const ordinal_view_t& vertClusters_,
            const in_rowmap_t& rowmap_,
            const in_entries_t& colinds_,
            const in_values_t& values_,
            const lno_t& numRows_,
            const mag_view_t& weights_) :
          clusterOffsets(clusterOffsets_),
          clusterVerts(clusterVerts_),
          vertClusters(vertClusters_),
          rowmap(rowmap_),
          colinds(colinds_),
          values(values_),
          numRows(numRows_),
          weights(weights_)
        {}

        KOKKOS_INLINE_FUNCTION void operator()(const team_member_t& t) const
        {
          const lno_t cluster = t.league_rank();
          const lno_t clusterBegin = clusterOffsets(cluster);
          const lno_t clusterEnd = clusterOffsets(cluster + 1);
          //Initialize weights for each vertex (NOT ordered within cluster, since that will change)
          //This is the absolute sum of off-diagonal matrix values corresponding to intra-cluster edges
          Kokkos::parallel_for(Kokkos::TeamThreadRange(t, clusterEnd - clusterBegin),
          [&](lno_t i)
          {
            const lno_t row = clusterVerts(clusterBegin + i);
            const size_type rowBegin = rowmap(row);
            const size_type rowEnd = rowmap(row + 1);
            mag_t w = 0;
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, rowEnd - rowBegin),
            [&](size_type j, mag_t& lweight)
            {
              const size_type ent = rowBegin + j;
              const lno_t col = colinds(ent);
              if(col < numRows && col != row && vertClusters(col) == cluster)
              {
                lweight += KAT::abs(values(ent));
              }
            }, w);
            Kokkos::single(Kokkos::PerThread(t),
            [&]()
            {
              weights(row) = w;
            });
          });
          t.team_barrier();
          //Until all vertices are reordered, swap the min-weighted vertex with the one
          //in position i (like selection sort)
          using MinLoc = Kokkos::MinLoc<mag_t, lno_t>;
          using MinLocVal = typename MinLoc::value_type;
          for(lno_t i = clusterBegin; i < clusterEnd - 1; i++)
          {
            //Find lowest-weight vertex (just on one thread)
            MinLocVal bestVert;
            bestVert.val = Kokkos::ArithTraits<mag_t>::max();
            //Whole team works on this loop
            Kokkos::parallel_reduce(Kokkos::TeamVectorRange(t, clusterEnd - i),
            [&](lno_t j, MinLocVal& lbest)
            {
              const lno_t row = clusterVerts(i + j);
              if(weights(row) < lbest.val)
              {
                lbest.val = weights(row);
                lbest.loc = i + j;
              }
            }, MinLoc(bestVert));
            //Swap the min-weighted row into position i
            Kokkos::single(Kokkos::PerTeam(t),
            [&]()
            {
              const lno_t temp = clusterVerts(i);
              clusterVerts(i) = clusterVerts(bestVert.loc);
              clusterVerts(bestVert.loc) = temp;
            });
            t.team_barrier();
            const lno_t elimRow = clusterVerts(i);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, clusterEnd - i - 1),
            [&](lno_t j)
            {
              const lno_t row = clusterVerts(i + j + 1);
              const size_type rowBegin = rowmap(row);
              const size_type rowEnd = rowmap(row + 1);
              //Compute the amount by which row's weight should be reduced
              mag_t w = 0;
              Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, rowEnd - rowBegin),
              [&](size_type k, mag_t& lweight)
              {
                size_type ent = rowBegin + k;
                lno_t col = colinds(ent);
                if(col == elimRow)
                  lweight += KAT::abs(values(ent));
              }, w);
              Kokkos::single(Kokkos::PerThread(t),
              [&]()
              {
                weights(row) -= w;
              });
            });
            t.team_barrier();
          }
        }

        //Cluster mapping
        ordinal_view_t clusterOffsets;
        ordinal_view_t clusterVerts;
        ordinal_view_t vertClusters;
        //Input matrix
        in_rowmap_t rowmap;
        in_entries_t colinds;
        in_values_t values;
        lno_t numRows;
        //Intra-cluster absolute sum of edge weights, per vertex
        mag_view_t weights;
      };

      void initialize_numeric()
      {
        auto gsHandle = get_gs_handle();
        if(!gsHandle->is_symbolic_called())
        {
          this->initialize_symbolic();
        }
        //Timer for whole numeric
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        Kokkos::Impl::Timer timer;
#endif
        size_type nnz = this->entries.extent(0);

        int suggested_vector_size = this->handle->get_suggested_vector_size(num_rows, nnz);
        int suggested_team_size = this->handle->get_suggested_team_size(suggested_vector_size);

        lno_t rows_per_team = this->handle->get_team_work_size(suggested_team_size, exec_space::concurrency(), num_rows);

        //Get the clusters back from handle
        ordinal_view_t clusterOffsets = gsHandle->get_cluster_xadj();
        ordinal_view_t clusterVerts = gsHandle->get_cluster_adj();
        ordinal_view_t vertClusters = gsHandle->get_vert_clusters();
        lno_t numClusters = clusterOffsets.extent(0) - 1;
        mag_view_t intraClusterWeights("Intra-cluster weights", num_rows);
        if(num_rows)
        {
          FlowOrderFunctor fof(clusterOffsets, clusterVerts, vertClusters, this->row_map, this->entries, this->values, num_rows, intraClusterWeights);
          lno_t fofTeamSize;
          {
            team_policy_t temp(numClusters, Kokkos::AUTO(), suggested_vector_size);
            fofTeamSize = temp.template team_size_recommended<FlowOrderFunctor>(fof, Kokkos::ParallelForTag());
            lno_t avgClusterSize = (num_rows + numClusters - 1) / numClusters;
            if(fofTeamSize > avgClusterSize)
              fofTeamSize = avgClusterSize;
          }
          Kokkos::parallel_for(team_policy_t(numClusters, fofTeamSize, suggested_vector_size), fof);
        }
        //Compute the compressed size of each cluster.
        offset_view_t streamOffsets(Kokkos::ViewAllocateWithoutInitializing("Matrix stream cluster offsets"), numClusters + 1);
        if(gsHandle->using_compact_scalars())
        {
          using Compression = ClusterCompression<true, HandleType, in_rowmap_t, in_entries_t, in_values_t, unit_t>;
          Kokkos::parallel_for(range_policy_t(0, numClusters), Compression::CompressedSizeFunctor(
                this->row_map, this->entries, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), streamOffsets));
          KokkosKernels::Impl::kk_exclusive_parallel_prefix_sum<offset_view_t, exec_space>(numClusters + 1, streamOffsets);
        }
        else
        {
          using Compression = ClusterCompression<false, HandleType, in_rowmap_t, in_entries_t, in_values_t, unit_t>;
          Kokkos::parallel_for(range_policy_t(0, numClusters), Compression::CompressedSizeFunctor(
                this->row_map, this->entries, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), streamOffsets));
          KokkosKernels::Impl::kk_exclusive_parallel_prefix_sum<offset_view_t, exec_space>(numClusters + 1, streamOffsets);
        }
        //Determine total compressed size, and allocate the data view
        auto compressedSizeHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(streamOffsets, numClusters));
        unit_view_t streamData(Kokkos::ViewAllocateWithoutInitializing("Matrix stream data"), compressedSizeHost());
        if(gsHandle->using_compact_scalars())
        {
          using Compression = ClusterCompression<true, HandleType, in_rowmap_t, in_entries_t, in_values_t, unit_t>;
          Kokkos::parallel_for(range_policy_t(0, numClusters), Compression::CompressFunctor(
                this->row_map, this->entries, this->values, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), streamOffsets, streamData));
        }
        else
        {
          using Compression = ClusterCompression<false, HandleType, in_rowmap_t, in_entries_t, in_values_t, unit_t>;
          Kokkos::parallel_for(range_policy_t(0, numClusters), Compression::CompressFunctor(
                this->row_map, this->entries, this->values, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), streamOffsets, streamData));
        }
        //Store compressed format in handle
        gsHandle->set_stream_offsets(streamOffsets);
        gsHandle->set_stream_data(streamData);
        gsHandle->set_call_numeric(true);
        exec_space().fence();
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
        std::cout << "NUMERIC:" << timer.seconds() << std::endl;
#endif
      }

      //Non-permute apply using range policy
      template<typename CompressionApply>
      void applyColorSetRange(
          const offset_view_t& streamOffsets, const unit_view_t& streamData,
          const typename CompressionApply::X_t& x,
          const typename CompressionApply::Y_t& y,
          lno_t colorSetBegin, lno_t colorSetEnd)
      {
        Kokkos::parallel_for(range_policy_t(colorSetBegin, colorSetEnd),
            typename CompressionApply::ApplyFunctor(streamOffsets, streamData, x, y, this->omega));
      }

      template <typename x_value_array_type, typename y_value_array_type>
      void apply(
          x_value_array_type x_lhs_output_vec,
          y_value_array_type y_rhs_input_vec,
          bool init_zero_x_vector = false,
          int numIter = 1,
          nnz_scalar_t omega = Kokkos::Details::ArithTraits<scalar_t>::one(),
          bool apply_forward = true,
          bool apply_backward = true,
          bool update_y_vector = true)
      {
        auto gsHandle = get_gs_handle();

        size_type nnz = entries.extent(0);
        ordinal_view_t color_adj = gsHandle->get_color_adj();
        host_ordinal_view_t h_color_xadj = gsHandle->get_color_xadj();

        color_t numColors = gsHandle->get_num_colors();

        //TODO: for permuted X/Y algorithms, do the permutations here.
        //If zeroing out X, only need to do that to the permuted version, not the original
        if(init_zero_x_vector){
          KokkosKernels::Impl::zero_vector<x_value_array_type, exec_space>(num_cols, x_lhs_output_vec);
        }

        if(update_y_vector)
        {
          //TODO: permute
        }
        //TODO: figure out if this is correct heuristic for team dimensions
/*
        int suggested_vector_size = this->handle->get_suggested_vector_size(num_rows, nnz);
        int suggested_team_size = this->handle->get_suggested_team_size(suggested_vector_size);

          lno_t rows_per_team = this->handle->get_team_work_size(suggested_team_size, MyExecSpace::concurrency(), num_rows);
          //Get clusters per team. Round down to favor finer granularity, since this is sensitive to load imbalance
          lno_t clusters_per_team = rows_per_team / gsHandle->get_cluster_size();
          if(clusters_per_team == 0)
            clusters_per_team = 1;
        */
        
        if(gsHandle->using_compact_scalars())
        {
          using 
        }
        else
        {
        }

template <bool compactScalar, typename HandleType, typename rowmap_t, typename entries_t, typename values_t, typename X_t, typename Y_t>
struct CompressedClusterApply
        if (apply_forward)
        {
          gs._is_backward = false;
          for (color_t i = 0; i < numColors; ++i){
            lno_t color_index_begin = h_color_xadj(i);
            lno_t color_index_end = h_color_xadj(i + 1);
            int overall_work = color_index_end - color_index_begin;// /256 + 1;
            gs._color_set_begin = color_index_begin;
            gs._color_set_end = color_index_end;
            Kokkos::parallel_for("KokkosSparse::GaussSeidel::Team_PSGS::forward",
                                 team_policy_t((overall_work + gs._clusters_per_team - 1) / gs._clusters_per_team, team_size, vec_size),
                                 gs);
            MyExecSpace().fence();
          }
        }
        if (apply_backward)
        {
          gs._is_backward = true;
          if (numColors > 0)
            for (color_t i = numColors - 1; ; --i) {
              lno_t color_index_begin = h_color_xadj(i);
              lno_t color_index_end = h_color_xadj(i + 1);
              lno_t overall_work = color_index_end - color_index_begin;// /256 + 1;
              gs._color_set_begin = color_index_begin;
              gs._color_set_end = color_index_end;
              Kokkos::parallel_for("KokkosSparse::GaussSeidel::Team_PSGS::forward",
                                   team_policy_t((overall_work + gs._clusters_per_team - 1) / gs._clusters_per_team, team_size, vec_size),
                                   gs);
              MyExecSpace().fence();
              if (i == 0){
                break;
              }
            }
        }
        MyExecSpace().fence();
      }

      template<typename TPSGS>
      void IterativeTeamPSGS(
          TPSGS& gs,
          color_t numColors,
          nnz_lno_persistent_work_host_view_t h_color_xadj,
          lno_t team_size,
          lno_t vec_size,
          int num_iteration,
          bool apply_forward,
          bool apply_backward)
      {
        for (int i = 0; i < num_iteration; ++i)
          this->DoTeamPSGS(gs, numColors, h_color_xadj, team_size, vec_size, apply_forward, apply_backward);
      }

      template<typename TPSGS>
      void DoTeamPSGS(
          TPSGS& gs, color_t numColors, nnz_lno_persistent_work_host_view_t h_color_xadj,
          lno_t team_size, lno_t vec_size,
          bool apply_forward,
          bool apply_backward)
      {
      }

      template<typename PSGS>
      void IterativePSGS(
          PSGS& gs,
          color_t numColors,
          nnz_lno_persistent_work_host_view_t h_color_xadj,
          int num_iteration,
          bool apply_forward,
          bool apply_backward)
      {
        for (int i = 0; i < num_iteration; ++i){
          this->DoPSGS(gs, numColors, h_color_xadj, apply_forward, apply_backward);
        }
      }

      template<typename PSGS>
      void DoPSGS(
          PSGS &gs, color_t numColors, nnz_lno_persistent_work_host_view_t h_color_xadj,
          bool apply_forward,
          bool apply_backward)
      {
        if (apply_forward){
          for (color_t i = 0; i < numColors; ++i){
            lno_t color_index_begin = h_color_xadj(i);
            lno_t color_index_end = h_color_xadj(i + 1);
            gs._color_set_begin = color_index_begin;
            gs._color_set_end = color_index_end;
            Kokkos::parallel_for ("KokkosSparse::GaussSeidel::PSGS::forward",
                Kokkos::RangePolicy<MyExecSpace, PSGS_ForwardTag>
                (0, color_index_end - color_index_begin), gs);
            MyExecSpace().fence();
          }
        }
        if (apply_backward && numColors){
          for (size_type i = numColors - 1; ; --i){
            lno_t color_index_begin = h_color_xadj(i);
            lno_t color_index_end = h_color_xadj(i + 1);
            gs._color_set_begin = color_index_begin;
            gs._color_set_end = color_index_end;
            Kokkos::parallel_for ("KokkosSparse::GaussSeidel::PSGS::backward",
                Kokkos::RangePolicy<MyExecSpace, PSGS_BackwardTag>
                (0, color_index_end - color_index_begin), gs);
            MyExecSpace().fence();
            if (i == 0){
              break;
            }
          }
        }
      }
    }; //class ClusterGaussSeidel
  } //namespace Impl
} //namespace KokkosSparse

#endif


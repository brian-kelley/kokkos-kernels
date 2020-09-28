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

namespace KokkosSparse {
namespace Impl {

template <typename HandleType>
class ClusterGaussSeidel
{
public:

  using CGSHandle       = typename HandleType::ClusterGaussSeidelHandleType;
  using GCHandle        = typename HandleType::GraphColoringHandleType;

  //Primitve types
  using size_type       = typename CGSHandle::size_type;
  using lno_t           = typename CGSHandle::lno_t;
  using scalar_t        = typename CGSHandle::scalar_t;
  using KAT             = Kokkos::Details::ArithTraits<scalar_t>;
  using mag_t           = typename KAT::mag_type;
  using unit_t          = typename CGSHandle::unit_t;
  using color_t         = typename HandleType::GraphColoringHandleType::color_t;

  //Device types
  using exec_space      = typename CGSHandle::exec_space;
  using mem_space       = typename CGSHandle::mem_space;
  using device_t        = Kokkos::Device<exec_space, mem_space>;
  using range_policy_t  = Kokkos::RangePolicy<exec_space>;
  using team_policy_t   = Kokkos::TeamPolicy<exec_space>;
  using team_member_t   = typename team_policy_t::member_type;

  //Views and containers
  using offset_view_t           = typename CGSHandle::offset_view_t;
  using unmanaged_offset_view_t = typename CGSHandle::unmanaged_offset_view_t;
  using ordinal_view_t          = typename CGSHandle::ordinal_view_t;
  using unmanaged_ordinal_view_t = typename CGSHandle::unmanaged_ordinal_view_t;
  using host_ordinal_view_t     = typename CGSHandle::host_ordinal_view_t;
  using scalar_view_t           = typename CGSHandle::scalar_view_t;
  using unmanaged_scalar_view_t = typename CGSHandle::unmanaged_scalar_view_t;
  using rowmajor_vector_t       = typename CGSHandle::rowmajor_vector_t;
  using colmajor_vector_t       = typename CGSHandle::colmajor_vector_t;
  using unit_view_t             = typename CGSHandle::unit_view_t;
  using mag_view_t              = Kokkos::View<mag_t*, mem_space>;
  using color_view_t            = typename HandleType::GraphColoringHandleType::color_view_t;
  using bitset_t                = Kokkos::Bitset<exec_space>;
  using const_bitset_t          = Kokkos::ConstBitset<exec_space>;

private:
  HandleType *handle;

  lno_t num_rows;
  lno_t num_cols;
  //The unmanaged row_map/entries can either point to the input graph, or symmetrized graph
  unmanaged_offset_view_t row_map;
  unmanaged_ordinal_view_t entries;
  unmanaged_scalar_view_t values;  //during symbolic, is empty
  //These are temporary (managed) views to store symmetrized graph during symbolic only.
  //Their lifetime is that of the ClusterGaussSeidel impl instance, not the handle.
  offset_view_t sym_row_map;
  ordinal_view_t sym_entries;
  //Whether the square part of the input matrix (1...#rows, 1...#rows) is symmetric
  bool is_symmetric;

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

public:

  /**
   * \brief constructor for symbolic
   */
  ClusterGaussSeidel(HandleType *handle_,
              lno_t num_rows_,
              lno_t num_cols_,
              const unmanaged_offset_view_t& row_map_,
              const unmanaged_ordinal_view_t& entries_,
              bool is_symmetric_ = true):
    handle(handle_),
    num_rows(num_rows_), num_cols(num_cols_),
    row_map(row_map_.data(), row_map_.extent(0)), entries(entries_.data(), entries_.extent(0)),
    values(),
    is_symmetric(is_symmetric_)
  {}

  /**
   * \brief constructor for numeric/apply (no longer care about structural symmetry)
   */
  ClusterGaussSeidel(HandleType *handle_,
              lno_t num_rows_,
              lno_t num_cols_,
              const unmanaged_offset_view_t& row_map_,
              const unmanaged_ordinal_view_t& entries_,
              const unmanaged_scalar_view_t& values_):
    handle(handle_), num_rows(num_rows_), num_cols(num_cols_),
    row_map(row_map_.data(), row_map_.extent(0)), entries(entries_.data(), entries_.extent(0)), values(values_),
    is_symmetric(true)
  {}

  //Functor to swap the numbers of two colors,
  //so that the last cluster has the last color.
  //Except, doesn't touch the color of the last cluster,
  //since that value is needed the entire time this is running.
  struct ClusterColorRelabelFunctor
  {
    ClusterColorRelabelFunctor(const color_view_t& colors_, color_t numClusterColors_, lno_t numClusters_)
      : colors(colors_), numClusterColors(numClusterColors_), numClusters(numClusters_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
    {
      if(colors(i) == numClusterColors)
        colors(i) = colors(numClusters - 1);
      else if(colors(i) == colors(numClusters - 1))
        colors(i) = numClusterColors;
    }

    color_view_t colors;
    color_t numClusterColors;
    lno_t numClusters;
  };

  //Relabel the last cluster, after running ClusterColorRelabelFunctor.
  //Call with a one-element range policy.
  struct RelabelLastColorFunctor
  {
    RelabelLastColorFunctor(const color_view_t& colors_, color_t numClusterColors_, lno_t numClusters_)
      : colors(colors_), numClusterColors(numClusterColors_), numClusters(numClusters_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_type) const
    {
      colors(numClusters - 1) = numClusterColors;
    }
    
    color_view_t colors;
    color_t numClusterColors;
    lno_t numClusters;
  };

  struct ClusterToVertexColoring
  {
    ClusterToVertexColoring(const color_view_t& clusterColors_, const color_view_t& vertexColors_, lno_t numRows_, lno_t numClusters_, lno_t clusterSize_)
      : clusterColors(clusterColors_), vertexColors(vertexColors_), numRows(numRows_), numClusters(numClusters_), clusterSize(clusterSize_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
    {
      size_type cluster = i / clusterSize;
      size_type clusterOffset = i - cluster * clusterSize;
      vertexColors(i) = ((clusterColors(cluster) - 1) * clusterSize) + clusterOffset + 1;
    }

    color_view_t clusterColors;
    color_view_t vertexColors;
    lno_t numRows;
    lno_t numClusters;
    lno_t clusterSize;
  };

  struct ClusterSizeFunctor
  {
    ClusterSizeFunctor(const ordinal_view_t& counts_, const ordinal_view_t& vertClusters_)
      : counts(counts_), vertClusters(vertClusters_)
    {}
    KOKKOS_INLINE_FUNCTION void operator()(const lno_t i) const
    {
      Kokkos::atomic_increment(&counts(vertClusters(i)));
    }
    ordinal_view_t counts;
    ordinal_view_t vertClusters;
  };

  struct ClusterBalanceStep1
  {
    ClusterBalanceStep1(const ordinal_view_t& tentVertClusters_, const ordinal_view_t& vertClusters_, const ordinal_view_t& clusterSizes_, const ordinal_view_t& clusterCounters_, lno_t targetSize_)
      : tentVertClusters(tentVertClusters_), vertClusters(vertClusters_), clusterSizes(clusterSizes_), clusterCounters(clusterCounters_), targetSize(targetSize_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t i) const
    {
      lno_t tentCluster = tentVertClusters(i);
      if(clusterSizes(tentCluster) <= targetSize + 1)
      {
        //Cluster is not overfull, so just keep the label
        Kokkos::atomic_increment(&clusterCounters(tentCluster));
        vertClusters(i) = tentCluster;
      }
    }

    ordinal_view_t tentVertClusters;
    ordinal_view_t vertClusters;
    ordinal_view_t clusterSizes;
    ordinal_view_t clusterCounters;
    lno_t targetSize;
  };

  struct ClusterBalanceStep2
  {
    ClusterBalanceStep2(const ordinal_view_t& tentVertClusters_, const ordinal_view_t& vertClusters_, const ordinal_view_t& clusterSizes_, const ordinal_view_t& clusterCounters_, const unmanaged_offset_view_t& rowmap_, const unmanaged_ordinal_view_t& entries_, lno_t targetSize_, lno_t numRows_)
      : tentVertClusters(tentVertClusters_), vertClusters(vertClusters_), clusterSizes(clusterSizes_), clusterCounters(clusterCounters_), rowmap(rowmap_), entries(entries_), targetSize(targetSize_), numRows(numRows_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t i) const
    {
      lno_t tentCluster = tentVertClusters(i);
      if(clusterSizes(tentCluster) > targetSize + 1)  //note: this condition is exactly disjoint with the one from Step1
      {
        //Tentative cluster is overfull, so attempt to copy a neighbor's label
        size_type rowBegin = rowmap(i);
        size_type rowEnd = rowmap(i + 1);
        lno_t newCluster = tentCluster;
        for(size_type j = rowBegin; j < rowEnd; j++)
        {
          lno_t nei = entries(j);
          if(nei >= numRows || nei == i)
            continue;
          lno_t neiCluster = tentVertClusters(nei);
          if(neiCluster == tentCluster || clusterSizes(neiCluster) > targetSize)
            continue;
          //Attempt to copy this label
          if(Kokkos::atomic_fetch_add(&clusterCounters(neiCluster), 1) <= targetSize)
          {
            newCluster = neiCluster;
            break;
          }
        }
        vertClusters(i) = newCluster;
      }
    }

    ordinal_view_t tentVertClusters;
    ordinal_view_t vertClusters;
    ordinal_view_t clusterSizes;
    ordinal_view_t clusterCounters;
    unmanaged_offset_view_t rowmap;
    unmanaged_ordinal_view_t entries;
    lno_t targetSize;
    lno_t numRows;
  };

  struct FillClusterVertsFunctor
  {
    FillClusterVertsFunctor(const ordinal_view_t& clusterOffsets_, const ordinal_view_t& clusterVerts_, const ordinal_view_t& vertClusters_, const ordinal_view_t& insertCounts_)
      : clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), vertClusters(vertClusters_), insertCounts(insertCounts_)
    {}
    KOKKOS_INLINE_FUNCTION void operator()(const lno_t i) const
    {
      lno_t cluster = vertClusters(i);
      lno_t offset = clusterOffsets(cluster) + Kokkos::atomic_fetch_add(&insertCounts(cluster), 1);
      clusterVerts(offset) = i;
    }
    ordinal_view_t clusterOffsets;
    ordinal_view_t clusterVerts;
    ordinal_view_t vertClusters;
    ordinal_view_t insertCounts;
  };

  struct BuildCrossClusterMaskFunctor
  {
    BuildCrossClusterMaskFunctor(const unmanaged_offset_view_t& rowmap_, const unmanaged_ordinal_view_t& entries_, const ordinal_view_t& clusterOffsets_, const ordinal_view_t& clusterVerts_, const ordinal_view_t& vertClusters_, const bitset_t& mask_)
      : numRows(rowmap_.extent(0) - 1), rowmap(rowmap_), entries(entries_), clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), vertClusters(vertClusters_), mask(mask_)
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
              lno_t nei = entries(rowmap(row) + j);
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
    unmanaged_offset_view_t rowmap;
    unmanaged_ordinal_view_t entries;
    ordinal_view_t clusterOffsets;
    ordinal_view_t clusterVerts;
    ordinal_view_t vertClusters;
    bitset_t mask;
  };

  struct FillClusterEntriesFunctor
  {
    FillClusterEntriesFunctor(
        const unmanaged_offset_view_t& rowmap_, const unmanaged_ordinal_view_t& entries_, const offset_view_t& clusterRowmap_, const ordinal_view_t& clusterEntries_, const ordinal_view_t& clusterOffsets_, const ordinal_view_t& clusterVerts_, const ordinal_view_t& vertClusters_, const bitset_t& edgeMask_)
      : rowmap(rowmap_), entries(entries_), clusterRowmap(clusterRowmap_), clusterEntries(clusterEntries_), clusterOffsets(clusterOffsets_), clusterVerts(clusterVerts_), vertClusters(vertClusters_), edgeMask(edgeMask_)
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
            clusterEntries(clusterEdge++) = vertClusters(entries(j));
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
    unmanaged_offset_view_t rowmap;
    unmanaged_ordinal_view_t entries;
    offset_view_t  clusterRowmap;
    ordinal_view_t clusterEntries;
    ordinal_view_t clusterOffsets;
    ordinal_view_t clusterVerts;
    ordinal_view_t vertClusters;
    const_bitset_t edgeMask;
  };

  void printClusteringStatistics(const ordinal_view_t& vc, lno_t numClusters)
  {
    double clusterStdev = 0;
    double clusterMean = (double) this->num_rows / numClusters;
    lno_t minCluster = this->num_rows;
    lno_t maxCluster = 0;
    auto vch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vc);
    std::vector<int> clusterSizes(numClusters, 0);
    for(lno_t i = 0; i < this->num_rows; i++)
      clusterSizes[vch(i)]++;
    for(lno_t i = 0; i < numClusters; i++)
    {
      lno_t cs = clusterSizes[i];
      if(cs > maxCluster)
        maxCluster = cs;
      if(cs < minCluster)
        minCluster = cs;
      double diff = clusterMean - cs;
      clusterStdev += diff * diff;
    }
    clusterStdev /= numClusters;
    clusterStdev = sqrt(clusterStdev);
    std::cout << "Mean and standard deviation of cluster size: " << ((double) this->num_rows / numClusters) << ", " << clusterStdev << '\n';
    std::cout << "Min, max cluster size: " << minCluster << ", " << maxCluster << '\n';
  }

  //Break up large clusters into clusters <= maxSize. Just take naturally ordered
  void serialFission(ordinal_view_t& vc, int& numClusters, int maxSize)
  {
    auto vch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vc);
    std::vector<int> clusterSizes(numClusters, 0);
    for(int i = 0; i < vch.extent(0); i++)
    {
      clusterSizes[vch(i)]++;
    }
    std::vector<int> fissionFactors(numClusters);
    for(int i = 0; i < numClusters; i++)
    {
      fissionFactors[i] = clusterSizes[i] / maxSize + 1;
    }
    std::vector<int> newLabelOffsets(numClusters);
    int accum = 0;
    for(int i = 0; i < numClusters; i++)
    {
      newLabelOffsets[i] = accum;
      accum += fissionFactors[i];
    }
    std::vector<int> counters(numClusters, 0);
    //New number of clusters
    numClusters = accum;
    //update labels
    for(int i = 0; i < vch.extent(0); i++)
    {
      int oldLabel = vch(i);
      int offset = newLabelOffsets[oldLabel];
      int oldSize = clusterSizes[oldLabel];
      vch(i) = offset + ((double) counters[oldLabel] / oldSize) * fissionFactors[oldLabel];
      counters[oldLabel]++;
    }
    Kokkos::deep_copy(vc, vch);
    std::cout << "Fission increased number of clusters from " << clusterSizes.size() << " to " << numClusters << '\n';
  }

  struct ClusterFissionOffsets
  {
    ClusterFissionOffsets(const ordinal_view_t& clusterSizes_, const ordinal_view_t& fissionOffsets_, lno_t maxSize_)
      : clusterSizes(clusterSizes_), fissionOffsets(fissionOffsets_), maxSize(maxSize_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i, lno_t& loffset, bool finalPass) const
    {
      lno_t origSize = clusterSizes(i);
      lno_t splittingFactor = origSize / maxSize + 1;
      if(finalPass)
      {
        fissionOffsets(i) = loffset;
        if(i == clusterSizes.extent(0) - 1)
          fissionOffsets(i + 1) = loffset + splittingFactor;
      }
      loffset += splittingFactor;
    }

    ordinal_view_t clusterSizes;
    ordinal_view_t fissionOffsets;
    lno_t maxSize;
  };

  struct ClusterFissionRelabel
  {
    ClusterFissionRelabel(const ordinal_view_t& vertClusters_, const ordinal_view_t& fissionOffsets_, const ordinal_view_t& clusterSizes_, const ordinal_view_t& counters_)
      : vertClusters(vertClusters_), fissionOffsets(fissionOffsets_), clusterSizes(clusterSizes_), counters(counters_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t i) const
    {
      lno_t oldLabel = vertClusters(i);
      lno_t counter = Kokkos::atomic_fetch_add(&counters(oldLabel), 1);
      lno_t beginOffset = fissionOffsets(oldLabel);
      lno_t splittingFactor = fissionOffsets(oldLabel + 1) - beginOffset;
      //clusterSizes(oldLabel) is split into splittingFactor different clusters, equally sized
      vertClusters(i) = beginOffset + ((float) counter / clusterSizes(oldLabel)) * splittingFactor;
    }

    ordinal_view_t vertClusters;
    ordinal_view_t fissionOffsets;
    ordinal_view_t clusterSizes;
    ordinal_view_t counters;
  };

  struct ClusterSizeComparator
  {
    ClusterSizeComparator(const ordinal_view_t& clusterSizes_)
      : clusterSizes(clusterSizes_)
    {}

    KOKKOS_INLINE_FUNCTION bool operator()(lno_t c1, lno_t c2) const
    {
      return clusterSizes(c1) > clusterSizes(c2);
    }

    ordinal_view_t clusterSizes;
  };

  void initialize_symbolic()
  {
    if(this->num_rows == 0)
      return;
    CGSHandle* gsHandle = get_gs_handle();
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
    Kokkos::Impl::Timer timer;
#endif
    //sym_row_map/sym_entries is only used here for clustering.
    //Create them as non-const, unmanaged views to avoid
    //duplicating a bunch of code between the
    //symmetric and non-symmetric input cases.
    if(!this->is_symmetric)
    {
      Kokkos::Timer t;
      KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap
        <unmanaged_offset_view_t, unmanaged_ordinal_view_t, offset_view_t, ordinal_view_t, exec_space>
        (this->num_rows, this->row_map, this->entries, this->sym_row_map, this->sym_entries);
      //no longer need the original graph
      //this is a mutable -> const conversion
      this->row_map = unmanaged_offset_view_t(sym_row_map.data(), sym_row_map.extent(0));
      this->entries = unmanaged_ordinal_view_t(sym_entries.data(), sym_entries.extent(0));
      std::cout << "*** Symmetrize graph time: " << t.seconds() << " s\n";
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
      std::cout << "SYMMETRIZING TIME: " << timer.seconds() << std::endl;
      timer.reset();
#endif
    }
    //Now that a symmetric graph is available, build the cluster graph (also symmetric)
    lno_t numClusters;
    ordinal_view_t clusterOffsets;
    ordinal_view_t clusterVerts(Kokkos::ViewAllocateWithoutInitializing("Cluster -> vertices"), this->num_rows);
    ordinal_view_t vertClusters;
    auto clusterAlgo = gsHandle->get_clustering_algo();
    if(clusterAlgo == CLUSTER_DEFAULT)
      clusterAlgo = CLUSTER_MIS2;
    switch(clusterAlgo)
    {
      case CLUSTER_MIS2:
      {
        //Raw MIS2 sometimes gives an imbalanced coarsening, enough to slow down apply performance. So run a balancing pass over it.
        vertClusters = KokkosGraph::Experimental::graph_mis2_coarsen<exec_space, unmanaged_offset_view_t, unmanaged_ordinal_view_t, ordinal_view_t>
          (this->row_map, this->entries, numClusters, KokkosGraph::MIS2_FAST);
        //This value for max size keeps all large cluster sizes between 2/3 and 4/3 of the original average
        Kokkos::Timer fissionTimer;
        ordinal_view_t clusterSizes("clusterSizes", numClusters);
        Kokkos::parallel_for(range_policy_t(0, this->num_rows), ClusterSizeFunctor(clusterSizes, vertClusters));
        lno_t newNumClusters;
        ordinal_view_t fissionOffsets(Kokkos::ViewAllocateWithoutInitializing("Fission offsets"), numClusters + 1);
        double maxClusterSize = 4.0 / 3.0 * this->num_rows / numClusters;
        Kokkos::parallel_scan(range_policy_t(0, numClusters), ClusterFissionOffsets(clusterSizes, fissionOffsets, maxClusterSize), newNumClusters);
        ordinal_view_t counters("Counters", numClusters);
        Kokkos::parallel_for(range_policy_t(0, this->num_rows), ClusterFissionRelabel(vertClusters, fissionOffsets, clusterSizes, counters));
        numClusters = newNumClusters;
        double ft = fissionTimer.seconds();
        std::cout << "*** Fission time: " << ft << " s\n";
        break;
      }
      case CLUSTER_BALLOON:
      {
        lno_t clusterSize = gsHandle->get_cluster_size();
        numClusters = (this->num_rows + clusterSize - 1) / clusterSize;
        BalloonClustering<HandleType, unmanaged_offset_view_t, unmanaged_ordinal_view_t, ordinal_view_t>
          balloon(this->num_rows, this->row_map, this->entries);
        vertClusters = balloon.run(clusterSize);
        break;
      }
      case CLUSTER_DEFAULT:
      {
        throw std::logic_error("Logic to choose default clustering algorithm never got called");
      }
      default:
        throw std::runtime_error("Clustering algo " + std::to_string((int) clusterAlgo) + " is not implemented");
    }
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
    std::cout << "Graph clustering: " << timer.seconds() << '\n';
    timer.reset();
#endif
    clusterOffsets = ordinal_view_t("Cluster offsets", numClusters + 1);
    //Construct the cluster offset and vertex array. These allow fast iteration over all vertices in a given cluster.
    Kokkos::parallel_for(range_policy_t(0, this->num_rows), ClusterSizeFunctor(clusterOffsets, vertClusters));
    KokkosKernels::Impl::exclusive_parallel_prefix_sum<ordinal_view_t, exec_space>(numClusters + 1, clusterOffsets);
    {
      ordinal_view_t tempInsertCounts("Temporary cluster insert counts", numClusters);
      Kokkos::parallel_for(range_policy_t(0, this->num_rows), FillClusterVertsFunctor(clusterOffsets, clusterVerts, vertClusters, tempInsertCounts));
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
    size_type nnz = this->entries.extent(0);
    int vectorSize = this->handle->get_suggested_vector_size(this->num_rows, nnz);
    bitset_t crossClusterEdgeMask(nnz);
    size_type numClusterEdges;
    {
      BuildCrossClusterMaskFunctor buildEdgeMask(this->row_map, this->entries,
          clusterOffsets, clusterVerts, vertClusters, crossClusterEdgeMask);
      int sharedPerTeam = buildEdgeMask.team_shmem_size(0); //using team-size = 0 for since no per-thread shared is used.
      int teamSize = KokkosKernels::Impl::get_suggested_team_size<team_policy_t>(
          buildEdgeMask, vectorSize, sharedPerTeam, 0);
      Kokkos::parallel_for(team_policy_t(numClusters, teamSize, vectorSize)
          .set_scratch_size(0, Kokkos::PerTeam(sharedPerTeam)), buildEdgeMask);
      numClusterEdges = crossClusterEdgeMask.count();
    }
    offset_view_t clusterRowmap(Kokkos::ViewAllocateWithoutInitializing("Cluster graph rowmap"), numClusters + 1);
    ordinal_view_t clusterEntries(Kokkos::ViewAllocateWithoutInitializing("Cluster graph entries"), numClusterEdges);
    Kokkos::parallel_scan(range_policy_t(0, this->num_rows), FillClusterEntriesFunctor
        (this->row_map, this->entries, clusterRowmap, clusterEntries, clusterOffsets, clusterVerts, vertClusters, crossClusterEdgeMask));
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
        for(size_type j = clusterRowmapHost(i); j < clusterRowmapHost(i + 1); j++)
        {
          printf("%d ", (int) clusterEntriesHost(j));
        }
        putchar('\n');
      }
      printf("\n\n\n");
    }
#endif
    //Get the coloring of the cluster graph.
    color_view_t colors;
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
    HandleType kh;
    kh.create_graph_coloring_handle(KokkosGraph::COLORING_DEFAULT);
    KokkosGraph::Experimental::graph_color_symbolic(&kh, numClusters, numClusters, clusterRowmap, clusterEntries);
    //retrieve colors
    auto coloringHandle = kh.get_graph_coloring_handle();
    colors = coloringHandle->get_vertex_colors();
    numColors = coloringHandle->get_num_colors();
    kh.destroy_graph_coloring_handle();
    std::cout << "# colors used for coarse graph " << numColors << '\n';
#endif
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
    std::cout << "Coloring: " << timer.seconds() << '\n';
    timer.reset();
#endif
    ordinal_view_t color_xadj; //offsets for color_sets
    ordinal_view_t color_adj;  //rows grouped by color
    KokkosKernels::Impl::create_reverse_map
      <typename HandleType::GraphColoringHandleType::color_view_t,
       ordinal_view_t, exec_space>
      (numClusters, numColors, colors, color_xadj, color_adj);
    //Get color set offsets on host
    host_ordinal_view_t color_xadj_host(Kokkos::ViewAllocateWithoutInitializing("Color xadj"), color_xadj.extent(0));
    Kokkos::deep_copy(color_xadj_host, color_xadj);
    //Within each color set, sort descending by cluster size.
    ordinal_view_t clusterSizes("Cluster sizes", numClusters);
    Kokkos::parallel_for(range_policy_t(0, this->num_rows), ClusterSizeFunctor(clusterSizes, vertClusters));
    ClusterSizeComparator compareByClusterSize(clusterSizes);
    for(int c = 0; c < numColors; c++)
    {
      auto colorSet = Kokkos::subview(color_adj, Kokkos::make_pair(color_xadj_host(c), color_xadj_host(c + 1)));
      KokkosKernels::Impl::bitonicSort<decltype(colorSet), exec_space, lno_t, ClusterSizeComparator>(colorSet, compareByClusterSize);
    }
    exec_space().fence();
#ifdef KOKKOSSPARSE_IMPL_TIME_REVERSE
    std::cout << "CREATE_REVERSE_MAP:" << timer.seconds() << std::endl;
    timer.reset();
#endif
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
        const unmanaged_offset_view_t& rowmap_,
        const unmanaged_ordinal_view_t& entries_,
        const unmanaged_scalar_view_t& values_,
        const lno_t numRows_,
        const mag_view_t& weights_) :
      clusterOffsets(clusterOffsets_),
      clusterVerts(clusterVerts_),
      vertClusters(vertClusters_),
      rowmap(rowmap_),
      entries(entries_),
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
          const lno_t col = entries(ent);
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
            lno_t col = entries(ent);
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
    unmanaged_offset_view_t rowmap;
    unmanaged_ordinal_view_t entries;
    unmanaged_scalar_view_t values;
    lno_t numRows;
    //Intra-cluster absolute sum of edge weights, per vertex
    mag_view_t weights;
  };

  //Scan functor to generate best permutation for apply
  //First groups by color, then cluster.
  //Note: requires flow ordering to be done already
  struct PermuteOrderFunctor
  {
    PermuteOrderFunctor(
        const ordinal_view_t& perm_, //output
        const ordinal_view_t& color_adj_,
        const ordinal_view_t& clusterOffsets_,
        const ordinal_view_t& clusterVerts_,
        const lno_t numRows_) :
      perm(perm_),
      color_adj(color_adj_),
      clusterOffsets(clusterOffsets_),
      clusterVerts(clusterVerts_),
      numRows(numRows_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(const lno_t i, lno_t& loffset, bool finalPass) const
    {
      const lno_t cluster = color_adj(i);
      const lno_t clusterBegin = clusterOffsets(cluster);
      const lno_t clusterSize = clusterOffsets(cluster + 1) - clusterBegin;
      if(finalPass)
      {
        //populate perm for this cluster.
        //loffset is where it starts, and clusterVerts gives the row list
        for(lno_t j = 0; j < clusterSize; j++)
        {
          lno_t v = clusterVerts(clusterBegin + j);
          perm(loffset + j) = v;
        }
      }
      loffset += clusterSize;
    }

    //Cluster mapping
    ordinal_view_t perm;
    ordinal_view_t color_adj;
    ordinal_view_t clusterOffsets;
    ordinal_view_t clusterVerts;
    lno_t numRows;
  };

  struct PermuteInverseFunctor
  {
    PermuteInverseFunctor(
        const ordinal_view_t& perm_,    //input
        const ordinal_view_t& invPerm_) //output
      : perm(perm_), invPerm(invPerm_)
    {}

    KOKKOS_INLINE_FUNCTION void operator()(lno_t i) const
    {
      invPerm(perm(i)) = i;
    }

    ordinal_view_t perm;
    ordinal_view_t invPerm;
  };

  void initialize_numeric()
  {
    if(this->num_rows == 0)
      return;
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

    int suggested_vector_size = this->handle->get_suggested_vector_size(this->num_rows, nnz);
    //int suggested_team_size = this->handle->get_suggested_team_size(suggested_vector_size);

    //lno_t rows_per_team = this->handle->get_team_work_size(suggested_team_size, exec_space::concurrency(), this->num_rows);
    //Get the clusters back from handle
    ordinal_view_t clusterOffsets = gsHandle->get_cluster_xadj();
    ordinal_view_t clusterVerts = gsHandle->get_cluster_adj();
    ordinal_view_t vertClusters = gsHandle->get_vert_clusters();
    lno_t numClusters = clusterOffsets.extent(0) - 1;
    mag_view_t intraClusterWeights("Intra-cluster weights", this->num_rows);
    if(this->num_rows)
    {
      FlowOrderFunctor fof(clusterOffsets, clusterVerts, vertClusters, this->row_map, this->entries, this->values, this->num_rows, intraClusterWeights);
      lno_t fofTeamSize;
      {
        team_policy_t temp(numClusters, Kokkos::AUTO(), suggested_vector_size);
        fofTeamSize = temp.template team_size_recommended<FlowOrderFunctor>(fof, Kokkos::ParallelForTag());
        lno_t avgClusterSize = (this->num_rows + numClusters - 1) / numClusters;
        if(fofTeamSize > avgClusterSize)
          fofTeamSize = avgClusterSize;
      }
      Kokkos::parallel_for(team_policy_t(numClusters, fofTeamSize, suggested_vector_size), fof);
    }
    //If using permuted apply, get the row permutation map in each direction (1-1 functions)
    ordinal_view_t permutation;     //a list of input rows.
    ordinal_view_t invPermutation;  //the permuted position of each input row.
    if(gsHandle->use_permutation())
    {
      permutation = ordinal_view_t(Kokkos::ViewAllocateWithoutInitializing("CGS Permutation"), this->num_rows);
      invPermutation = ordinal_view_t(Kokkos::ViewAllocateWithoutInitializing("CGS Permutation^-1"), this->num_rows);
      Kokkos::parallel_scan(range_policy_t(0, numClusters),
          PermuteOrderFunctor(permutation, gsHandle->get_color_adj(), clusterOffsets, clusterVerts, this->num_rows));
      gsHandle->set_apply_permutation(permutation);
      Kokkos::parallel_for(range_policy_t(0, this->num_rows), PermuteInverseFunctor(permutation, invPermutation));
    }
    //Compute the compressed size of each cluster.
    offset_view_t streamOffsets(Kokkos::ViewAllocateWithoutInitializing("Matrix stream cluster offsets"), numClusters + 1);
    if(gsHandle->use_compact_scalars())
    {
      using Compression = ClusterCompression<true, CGSHandle>;
      if(gsHandle->use_permutation())
      {
        Kokkos::parallel_for(range_policy_t(0, numClusters), typename Compression::PermutedCompressedSizeFunctor(
              this->row_map, this->entries, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), streamOffsets));
      }
      else
      {
        Kokkos::parallel_for(range_policy_t(0, numClusters), typename Compression::CompressedSizeFunctor(
              this->row_map, this->entries, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), streamOffsets));
      }
    }
    else
    {
      using Compression = ClusterCompression<false, CGSHandle>;
      if(gsHandle->use_permutation())
      {
        Kokkos::parallel_for(range_policy_t(0, numClusters), typename Compression::PermutedCompressedSizeFunctor(
              this->row_map, this->entries, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), streamOffsets));
      }
      else
      {
        Kokkos::parallel_for(range_policy_t(0, numClusters), typename Compression::CompressedSizeFunctor(
              this->row_map, this->entries, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), streamOffsets));
      }
    }
    KokkosKernels::Impl::kk_exclusive_parallel_prefix_sum<offset_view_t, exec_space>(numClusters + 1, streamOffsets);
    //Determine total compressed size, and allocate the data view
    auto compressedSizeHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(streamOffsets, numClusters));
    unit_view_t streamData(Kokkos::ViewAllocateWithoutInitializing("Matrix stream data"), compressedSizeHost());
    if(gsHandle->use_compact_scalars())
    {
      using Compression = ClusterCompression<true, CGSHandle>;
      if(gsHandle->use_permutation())
      {
        Kokkos::parallel_for(range_policy_t(0, numClusters), typename Compression::PermutedCompressFunctor(
              this->row_map, this->entries, this->values, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), invPermutation, streamOffsets, streamData));
      }
      else
      {
        Kokkos::parallel_for(range_policy_t(0, numClusters), typename Compression::CompressFunctor(
              this->row_map, this->entries, this->values, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), streamOffsets, streamData));
      }
    }
    else
    {
      using Compression = ClusterCompression<false, CGSHandle>;
      if(gsHandle->use_permutation())
      {
        Kokkos::parallel_for(range_policy_t(0, numClusters), typename Compression::PermutedCompressFunctor(
              this->row_map, this->entries, this->values, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), invPermutation, streamOffsets, streamData));
      }
      else
      {
        Kokkos::parallel_for(range_policy_t(0, numClusters), typename Compression::CompressFunctor(
              this->row_map, this->entries, this->values, clusterOffsets, clusterVerts, gsHandle->get_color_adj(), streamOffsets, streamData));
      }
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

  //3 different modes for pre-apply vector
  //permutation/zero-initialization.
  //Just permute X: if !init_zero_x_vector && !update_y_vector
  struct Pre_Permute_X_Tag {};
  //Permute both: if !init_zero_x_vector && update_y_vector
  struct Pre_Permute_XY_Tag {};
  //Just permute Y: if init_zero_x_vector && update_y_vector
  struct Pre_Permute_Y_Tag {};
  //Final case is init_zero_x_vector && !update_y_vector,
  //so nothing gets permuted.

  template<typename X_t, typename Y_t>
  struct PrePermute
  {
    //Version that does both x and y
    PrePermute(const ordinal_view_t& permutation_, const X_t& orig_x_, const colmajor_vector_t& perm_x_, const Y_t& orig_y_, const rowmajor_vector_t& perm_y_)
      : permutation(permutation_), orig_x(orig_x_), perm_x(perm_x_), orig_y(orig_y_), perm_y(perm_y_),
      numRows(orig_y.extent(0)), numVecs(orig_y.extent(1))
    {}

    KOKKOS_FORCEINLINE_FUNCTION void permXRow(lno_t row) const
    {
      if(row < numRows)
      {
        //must be permuted
        lno_t origRow = permutation(row);
        for(lno_t col = 0; col < numVecs; col++)
          perm_x(row, col) = orig_x(origRow, col);
      }
      else
      {
        //copy to same posiiton
        for(lno_t col = 0; col < numVecs; col++)
          perm_x(row, col) = orig_x(row, col);
      }
    }

    KOKKOS_FORCEINLINE_FUNCTION void permYRow(lno_t row) const
    {
      lno_t origRow = permutation(row);
      for(lno_t col = 0; col < numVecs; col++)
        perm_y(row, col) = orig_y(origRow, col);
    }

    //Call over range (0, numCols)
    KOKKOS_INLINE_FUNCTION void operator()(Pre_Permute_X_Tag, lno_t row) const
    {
      permXRow(row);
    }

    //Call over range (0, numRows)
    KOKKOS_INLINE_FUNCTION void operator()(Pre_Permute_Y_Tag, lno_t row) const
    {
      permYRow(row);
    }

    //Call over range (0, numRows + numCols)
    KOKKOS_INLINE_FUNCTION void operator()(Pre_Permute_XY_Tag, lno_t i) const
    {
      if(i < numRows)
        permYRow(i);
      else
        permXRow(i - numRows);
    }

    ordinal_view_t permutation; //list of rows
    X_t orig_x;
    colmajor_vector_t perm_x;
    Y_t orig_y;
    rowmajor_vector_t perm_y;
    lno_t numRows;
    lno_t numVecs;
  };

  template<typename X_t>
  struct PostPermute
  {
    PostPermute(const ordinal_view_t& permutation_, const colmajor_vector_t& perm_x_, const X_t& orig_x_)
      : permutation(permutation_), perm_x(perm_x_), orig_x(orig_x_), numVecs(orig_x.extent(1))
    {}

    //Run this over range (0, numRows). Elements numRows...numCols
    //weren't modified and don't need to be copied back.
    KOKKOS_INLINE_FUNCTION void operator()(lno_t row) const
    {
      //A reversal of the pre-permute assignment.
      //This works because rows 0...numRows are permuted in a 1-1 mapping.
      lno_t origRow = permutation(row);
      for(lno_t col = 0; col < numVecs; col++)
        orig_x(origRow, col) = perm_x(row, col);
    }

    ordinal_view_t permutation; //list of rows
    colmajor_vector_t perm_x;
    X_t orig_x;
    lno_t numVecs;
  };

  //Non-permuted apply using range policy
  template<typename CompressionApply, bool permuted>
  void applyRange(
      color_t numColors, const host_ordinal_view_t& colorOffsets,
      const offset_view_t& streamOffsets, const unit_view_t& streamData,
      const typename CompressionApply::X_t& x,
      const typename CompressionApply::Y_t& y,
      bool isForward,
      scalar_t omega)
  {
    using ApplyFunctor = typename std::conditional<permuted,
    typename CompressionApply::PermuteApplyRange,
      typename CompressionApply::ApplyRange>::type;
    if(x.extent(1) == 1)
    {
      for(color_t i = 0; i < numColors; i++)
      {
        color_t c = isForward ? i : (numColors - 1 - i);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<exec_space, CGS_SingleVec_Tag>(colorOffsets(c), colorOffsets(c + 1)),
            ApplyFunctor(streamOffsets, streamData, x, y, omega));
      }
    }
    else
    {
      for(color_t i = 0; i < numColors; i++)
      {
        color_t c = isForward ? i : (numColors - 1 - i);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<exec_space, CGS_MultiVec_Tag>(colorOffsets(c), colorOffsets(c + 1)),
            ApplyFunctor(streamOffsets, streamData, x, y, omega));
      }
    }
  }

  //Non-permuted apply using team policy
  template<typename CompressionApply, bool permuted>
  void applyTeam(
      color_t numColors, const host_ordinal_view_t& colorOffsets,
      const offset_view_t& streamOffsets, const unit_view_t& streamData,
      const typename CompressionApply::X_t& x,
      const typename CompressionApply::Y_t& y,
      bool isForward,
      scalar_t omega)
  {
    using ApplyFunctor = typename std::conditional<permuted,
      typename CompressionApply::PermuteApplyTeam,
      typename CompressionApply::ApplyTeam>::type;
    //NOTE: unlike Range-based apply, Team uses a manually unrolled loop over vectors.
    //So the 1-vector case is still optimal, using the same functor
    ApplyFunctor f(streamOffsets, streamData, x, y, omega);
    //Decide the best team, thread size
    int threadSize = KokkosKernels::Impl::kk_get_suggested_vector_size(
        num_rows, entries.extent(0), KokkosKernels::Impl::kk_get_exec_space_type<exec_space>());
    int teamSize = KokkosKernels::Impl::get_suggested_team_size<team_policy_t, ApplyFunctor>(f, threadSize);
    for(color_t i = 0; i < numColors; i++)
    {
      color_t c = isForward ? i : (numColors - 1 - i);
      f.colorSetBegin = colorOffsets(c);
      f.colorSetEnd = colorOffsets(c + 1);
      int leagueSize = (f.colorSetEnd - f.colorSetBegin + teamSize - 1) / teamSize;
      Kokkos::parallel_for(team_policy_t(leagueSize, teamSize, threadSize), f);
    }
  }

  template<typename X_t, typename Y_t>
  void generalNonPermutedApply(CGSHandle* gsHandle, color_t numColors, const host_ordinal_view_t& colorOffsets, const offset_view_t& streamOffsets, const unit_view_t& streamData, const X_t& x, const Y_t& y, bool forward, scalar_t omega)
  {
    if(gsHandle->use_compact_scalars())
    {
      using CompressedApply = CompressedClusterApply<true, CGSHandle, X_t, Y_t>;
      if(gsHandle->use_teams())
      {
        applyTeam<CompressedApply, false>(numColors, colorOffsets, streamOffsets, streamData, x, y, forward, omega);
      }
      else
      {
        applyRange<CompressedApply, false>(numColors, colorOffsets, streamOffsets, streamData, x, y, forward, omega);
      }
    }
    else
    {
      using CompressedApply = CompressedClusterApply<false, CGSHandle, X_t, Y_t>;
      if(gsHandle->use_teams())
      {
        applyTeam<CompressedApply, false>(numColors, colorOffsets, streamOffsets, streamData, x, y, forward, omega);
      }
      else
      {
        applyRange<CompressedApply, false>(numColors, colorOffsets, streamOffsets, streamData, x, y, forward, omega);
      }
    }
  }

  void generalPermutedApply(CGSHandle* gsHandle, color_t numColors, const host_ordinal_view_t& colorOffsets, const offset_view_t& streamOffsets, const unit_view_t& streamData, const colmajor_vector_t& perm_x, const rowmajor_vector_t& perm_y, bool forward, scalar_t omega)
  {
    if(gsHandle->use_compact_scalars())
    {
      using CompressedApply = CompressedClusterApply<true, CGSHandle, colmajor_vector_t, rowmajor_vector_t>;
      if(gsHandle->use_teams())
      {
        applyTeam<CompressedApply, true>(numColors, colorOffsets, streamOffsets, streamData, perm_x, perm_y, forward, omega);
      }
      else
      {
        applyRange<CompressedApply, true>(numColors, colorOffsets, streamOffsets, streamData, perm_x, perm_y, forward, omega);
      }
    }
    else
    {
      using CompressedApply = CompressedClusterApply<false, CGSHandle, colmajor_vector_t, rowmajor_vector_t>;
      if(gsHandle->use_teams())
      {
        applyTeam<CompressedApply, true>(numColors, colorOffsets, streamOffsets, streamData, perm_x, perm_y, forward, omega);
      }
      else
      {
        applyRange<CompressedApply, true>(numColors, colorOffsets, streamOffsets, streamData, perm_x, perm_y, forward, omega);
      }
    }
  }

  template <typename X_t, typename Y_t>
  void apply(
      X_t x,
      Y_t y,
      bool init_zero_x_vector = false,
      int numIter = 1,
      scalar_t omega = KAT::one(),
      bool apply_forward = true,
      bool apply_backward = true,
      bool update_y_vector = true)
  {
    if(this->num_rows == 0)
    {
      if(init_zero_x_vector)
        Kokkos::deep_copy(x, KAT::zero());
      return;
    }
    auto gsHandle = get_gs_handle();
    colmajor_vector_t perm_x;
    rowmajor_vector_t perm_y;
    ordinal_view_t permutation;
    if(gsHandle->use_permutation())
    {
      //Lazily allocate permuted x/y to the right size (costs nothing if already the right size)
      gsHandle->allocate_perm_xy(y.extent(0), x.extent(0), x.extent(1));
      gsHandle->get_perm_xy(perm_x, perm_y);
      permutation = gsHandle->get_apply_permutation();
      PrePermute<X_t, Y_t> pf(permutation, x, perm_x, y, perm_y);
      if(!update_y_vector && !init_zero_x_vector)
      {
        //Only permute x.
        Kokkos::parallel_for(Kokkos::RangePolicy<exec_space, Pre_Permute_X_Tag>(0, this->num_cols), pf);
      }
      else if(update_y_vector && init_zero_x_vector)
      {
        //Only permute y.
        Kokkos::parallel_for(Kokkos::RangePolicy<exec_space, Pre_Permute_Y_Tag>(0, this->num_cols), pf);
      }
      else if(update_y_vector && !init_zero_x_vector)
      {
        //Permute both x and y.
        Kokkos::parallel_for(Kokkos::RangePolicy<exec_space, Pre_Permute_XY_Tag>(0, this->num_rows + this->num_cols), pf);
      }
    }
    if(init_zero_x_vector)
    {
      //zero out the entire x vector used for apply.
      if(gsHandle->use_permutation())
        KokkosKernels::Impl::zero_vector<colmajor_vector_t, exec_space>(this->num_cols, perm_x);
      else
        KokkosKernels::Impl::zero_vector<X_t, exec_space>(this->num_cols, x);
    }
    host_ordinal_view_t colorOffsets = gsHandle->get_color_xadj();
    color_t numColors = gsHandle->get_num_colors();
    auto streamOffsets = gsHandle->get_stream_offsets();
    auto streamData = gsHandle->get_stream_data();
    for(int iter = 0; iter < numIter; iter++)
    {
      if(apply_forward)
      {
        if(gsHandle->use_permutation())
        {
          generalPermutedApply(gsHandle, numColors, colorOffsets, streamOffsets, streamData, perm_x, perm_y, true, omega);
        }
        else
        {
          generalNonPermutedApply<X_t, Y_t>(gsHandle, numColors, colorOffsets, streamOffsets, streamData, x, y, true, omega);
        }
      }
      if(apply_backward)
      {
        if(gsHandle->use_permutation())
        {
          generalPermutedApply(gsHandle, numColors, colorOffsets, streamOffsets, streamData, perm_x, perm_y, false, omega);
        }
        else
        {
          generalNonPermutedApply<X_t, Y_t>(gsHandle, numColors, colorOffsets, streamOffsets, streamData, x, y, false, omega);
        }
      }
    }
    if(gsHandle->use_permutation())
    {
      //scatter the modified entries of perm_x back to x
      Kokkos::parallel_for(Kokkos::RangePolicy<exec_space>(0, this->num_rows),
          PostPermute<X_t>(permutation, perm_x, x));
      if(init_zero_x_vector && this->num_cols > this->num_rows)
      {
        //Also need to zero out rows num_rows...num_cols in input x.
        //Before, only perm_x was zeroed.
        auto remoteX = Kokkos::subview(x, std::make_pair(this->num_rows, this->num_cols), Kokkos::ALL());
        KokkosKernels::Impl::zero_vector<decltype(remoteX), exec_space>(this->num_cols - this->num_rows, remoteX);
      }
    }
    exec_space().fence();
  }
}; //class ClusterGaussSeidel

}} //namespace KokkosSparse::Impl

#endif


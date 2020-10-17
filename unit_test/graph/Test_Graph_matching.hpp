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

#include <gtest/gtest.h>
#include <random>
#include <Kokkos_Core.hpp>

#include "KokkosGraph_Matching.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosKernels_Handle.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

using namespace KokkosGraph;
using namespace KokkosGraph::Experimental;

namespace Test {

template<typename lno_t, typename size_type, typename rowmap_t, typename entries_t, typename matches_t>
void verifyMatching(
    lno_t numVerts,
    const rowmap_t& rowmap, const entries_t& entries,
    const matches_t& matches)
{
  //A maximal matching requires:
  //  -Every vertex is matched to at most one neighbor
  //  -No edge has two unmatched endpoints (maximal)
  std::vector<lno_t> numMatches(numVerts);
  std::vector<bool> isMatched(numVerts);
  for(lno_t i = 0; i < numVerts; i++)
  {
    if(matches(i) != i)
    {
      numMatches[matches(i)]++;
      isMatched[i] = true;
      isMatched[matches(i)] = true;
    }
  }
  for(lno_t i = 0; i < numVerts; i++)
  {
    EXPECT_LE(numMatches[i], 1) << "More than 1 vertex was matched with " << i;
  }
  for(lno_t i = 0; i < numVerts; i++)
  {
    for(size_type j = rowmap(i); j < rowmap(i + 1); j++)
    {
      lno_t nei = entries(j);
      if(nei <= i || nei >= numVerts)
       continue; 
      EXPECT_TRUE(isMatched[i] || isMatched[nei]) << "Neither endpoint of edge <" << i << ", " << nei << "> is matched, so not maximal";
    }
  }
}
}

template<typename scalar_unused, typename lno_t, typename size_type, typename device>
void test_matching(lno_t numVerts, size_type nnz, lno_t bandwidth, lno_t row_size_variance)
{
  using execution_space = typename device::execution_space;
  using crsMat = KokkosSparse::CrsMatrix<double, lno_t, device, void, size_type>;
  using graph_type = typename crsMat::StaticCrsGraphType;
  using c_rowmap_t = typename graph_type::row_map_type;
  using c_entries_t = typename graph_type::entries_type;
  using rowmap_t = typename c_rowmap_t::non_const_type;
  using entries_t = typename c_entries_t::non_const_type;
  //Generate graph, and add some out-of-bounds columns
  crsMat A = KokkosKernels::Impl::kk_generate_sparse_matrix<crsMat>(numVerts, numVerts, nnz, row_size_variance, bandwidth);
  auto G = A.graph;
  //Symmetrize the graph
  rowmap_t symRowmap;
  entries_t symEntries;
  KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap
    <c_rowmap_t, c_entries_t,
    rowmap_t, entries_t, execution_space>
      (numVerts, G.row_map, G.entries, symRowmap, symEntries);
  auto rowmapHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), symRowmap);
  auto entriesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), symEntries);
  auto matching = graph_match<device, rowmap_t, entries_t>(symRowmap, symEntries);
  auto matchingHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matching);
  Test::verifyMatching
    <lno_t, size_type, decltype(rowmapHost), decltype(entriesHost), decltype(matchingHost)>
    (numVerts, rowmapHost, entriesHost, matchingHost);
}

#define EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE) \
    TEST_F(TestCategory, graph##_##graph_matching##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) \
    { \
      test_matching<SCALAR, ORDINAL, OFFSET, DEVICE>(5000, 5000 * 20, 1000, 10); \
      test_matching<SCALAR, ORDINAL, OFFSET, DEVICE>(50, 50 * 10, 40, 10); \
      test_matching<SCALAR, ORDINAL, OFFSET, DEVICE>(5, 5 * 3, 5, 0); \
    }

#if defined(KOKKOSKERNELS_INST_DOUBLE)
#if(defined(KOKKOSKERNELS_INST_ORDINAL_INT) && defined(KOKKOSKERNELS_INST_OFFSET_INT)) \
  || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_TEST(double, int, int, TestExecSpace)
#endif

#if(defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && defined(KOKKOSKERNELS_INST_OFFSET_INT)) \
  || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_TEST(double, int64_t, int, TestExecSpace)
#endif

#if(defined(KOKKOSKERNELS_INST_ORDINAL_INT) && defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) \
  || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_TEST(double, int, size_t, TestExecSpace)
#endif

#if(defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) \
  || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_TEST(double, int64_t, size_t, TestExecSpace)
#endif
#endif

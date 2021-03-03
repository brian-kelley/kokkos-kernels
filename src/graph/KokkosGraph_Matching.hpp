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

#ifndef _KOKKOSGRAPH_MATCHING_HPP
#define _KOKKOSGRAPH_MATCHING_HPP

#include "KokkosGraph_Matching_impl.hpp"

namespace KokkosGraph{
namespace Experimental{

// Compute a distance-2 maximal independent set, given a symmetric CRS graph.
// Returns a list of the vertices in the set.
//
// Column indices >= num_verts are ignored.

template <typename device_t, typename rowmap_t, typename colinds_t, typename labels_t = typename colinds_t::non_const_type>
labels_t
graph_match(const rowmap_t& rowmap, const colinds_t& colinds)
{
  if(rowmap.extent(0) <= 1)
  {
    //there are no vertices to label
    return labels_t();
  }
  Impl::MaximalMatching<device_t, rowmap_t, colinds_t, labels_t> matching(rowmap, colinds);
  matching.compute();
  return matching.matches;
}

/*
//Run matching-based coarsening for numSteps iterations, producing coarsening labels with clusters of max size 2^numSteps.
template <typename device_t, typename rowmap_t, typename colinds_t, typename labels_t = typename colinds_t::non_const_type>
labels_t
graph_match_coarsen(const rowmap_t& rowmap, const colinds_t& colinds, int numSteps, typename colinds_t::non_const_value_type& numClusters)
{
  //Basic idea:
  //
  //Let L' = identity labels (iota)
  //For each step:
  //  Run matching on G
  //  Compress labels of G to form valid coarsening labels L
  //  Let L' = L(L')
  //  G := explicit coarsening of G with the labels
  //return L'
  using MMC = Impl::MaximalMatchCoarsening<device_t, rowmap_t, colinds_t, labels_t>;
  if(rowmap.extent(0) <= 1)
  {
    //there are no vertices to label
    return labels_t();
  }
  if(numSteps == 0)
  {
    labels_t l(Kokkos::ViewAllocateWithoutInitializing("CoarseLabels"), rowmap.extent(0) - 1);
    KokkosKernels::Impl::sequential_fill(l);
    return l;
  }
  MMC mc(rowmap, colinds);
  return mc.compute(numSteps, numClusters);
}
*/

}  // end namespace Experimental
}  // end namespace KokkosGraph

#endif

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

#ifndef KOKKOSGRAPH_BFS_H
#define KOKKOSGRAPH_BFS_H

//TODO: ETI the implementation

#include "impl/KokkosGraph_BFS_impl.hpp"

namespace KokkosGraph {
namespace Experimental {

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
}}

#endif


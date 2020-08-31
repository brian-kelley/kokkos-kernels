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

#ifndef _KOKKOSGRAPH_DISTANCE2_MIS_HPP
#define _KOKKOSGRAPH_DISTANCE2_MIS_HPP

#include "KokkosGraph_Distance2MIS_impl.hpp"

namespace KokkosGraph{

enum MIS2_Algorithm
{
  MIS2_QUALITY,
  MIS2_FAST
};

namespace Experimental{

// Compute a distance-2 maximal independent set, given a symmetric CRS graph.
// Returns a list of the vertices in the set.
//
// Column indices >= num_verts are ignored.

template <typename device_t, typename rowmap_t, typename colinds_t>
typename colinds_t::non_const_type
graph_d2_mis(const rowmap_t& rowmap, const colinds_t& colinds, MIS2_Algorithm algo = MIS2_FAST)
{
  if(rowmap.extent(0) <= 1)
  {
    //zero vertices means the MIS is empty.
    return typename colinds_t::non_const_type();
  }
  switch(algo)
  {
    case MIS2_QUALITY:
    {
      Impl::D2_MIS_ECL<device_t, rowmap_t, colinds_t> mis(rowmap, colinds);
      return mis.compute();
    }
    case MIS2_FAST:
    {
      Impl::D2_MIS_Luby<device_t, rowmap_t, colinds_t> mis(rowmap, colinds);
      return mis.compute();
    }
  }
  throw std::invalid_argument("graph_d2_mis: invalid algorithm");
}

}  // end namespace Experimental
}  // end namespace KokkosGraph

#endif

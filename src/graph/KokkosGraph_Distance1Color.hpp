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
#ifndef _KOKKOSGRAPH_DISTANCE1_COLOR_HPP
#define _KOKKOSGRAPH_DISTANCE1_COLOR_HPP

#include <sstream>

#include "KokkosGraph_Distance1ColorHandle.hpp"
#include "KokkosGraph_Distance1Color_impl.hpp"
#include "KokkosKernels_Utils.hpp"

namespace KokkosGraph{

namespace Experimental{

/**
 * Coloring an undirected graph, so that no adjacent pair of vertices has the same color.
 *
 * @param[in]  handle       The Kernel Handle
 * @param[in]  num_verts    Number of vertices to color (should match number of rows in the adjacency matrix)
 * @param[in]  unused
 * @param[in]  row_map      Row map of adjacency matrix
 * @param[in]  row_entries  Row entries of adjacency matrix
 * @param[in]  is_symmetric Whether the input matrix is known to be symmetric.
 *                          Note that "remote columns" (entries >= num_verts) are ignored and do not
 *                          affect symmetry.
 *
 *                          If the graph is not symmetric on input, it must be symmetrized internally,
 *                          which costs additional time (typically much less than the coloring itself)
 *                          and memory (up to twice as much space as the input graph).
 *
 *                          This is necessary so that the algorithm can guarantee forward progress: 
 *                          if U and V are adjacent in the undirected graph but it is not known whether
 *                          (U,V), (V,U) or both appear in the adjacency matrix, then it is not clear which
 *                          vertex should be uncolored during conflict detection. If both vertices are
 *                          uncolored, they may conflict in the next iteration, and so on indefinitely.
 *                          
 * @param[in]  is_sorted    Whether the provided adjacency matrix is known to be sorted within rows. 
 *                          In each row, "remote columns" (entries >= num_verts) must be placed after
 *                          "local columns", as is generally the case with Tpetra column maps.
 *
  \post handle->get_graph_coloring_handle()->get_vertex_colors() will return a view of length num_verts, containing the colors.
 */

template <class KernelHandle, typename lno_row_view_t_, typename lno_nnz_view_t_>
void graph_color(
    KernelHandle *handle,
    typename KernelHandle::nnz_lno_t num_verts,
    typename KernelHandle::nnz_lno_t,
    lno_row_view_t_ row_map,
    lno_nnz_view_t_ entries,
    bool is_symmetric = true,
    bool is_sorted = false)
{
  using InternalRowmap = typename lno_row_view_t_::non_const_type;
  using InternalEntries = typename lno_nnz_view_t_::non_const_type;
  InternalRowmap internalRowmap;
  InternalEntries internalEntries;
  Kokkos::Impl::Timer timer;
  if(is_symmetric)
  {
    internalRowmap = row_map;
    internalEntries = entries;
  }
  else
  {
    //Note: since this is done after the construction of "timer", it will
    //be included in overall_coloring_time.
    //TODO: make this its own timing section in the handle?
    using ExecSpace = typename KernelHandle::HandleExecSpace;
    KokkosKernels::Impl::symmetrize_graph_symbolic_hashmap<
      <lno_row_view_t_, lno_nnz_view_t_, InternalRowmap, InternalEntries, ExecSpace>
      (num_verts, row_map, entries, internalRowmap, internalEntries);
    //Being sorted doesn't help if the graph isn't symmetric
    is_sorted = false;
  }

  typename KernelHandle::GraphColoringHandleType *gch = handle->get_graph_coloring_handle();

  ColoringAlgorithm algorithm = gch->get_coloring_algo_type();

  typedef typename KernelHandle::GraphColoringHandleType::color_view_t color_view_type;

  gch->set_tictoc(handle->get_verbose());

  color_view_type colors_out = color_view_type("Graph Colors", num_rows);

  typedef typename Impl::GraphColor
      <typename KernelHandle::GraphColoringHandleType, InternalRowmap, InternalEntries> BaseGraphColoring;
  BaseGraphColoring *gc = NULL;

  switch (algorithm){
  case COLORING_SERIAL:
    gc = new BaseGraphColoring(num_rows, internalRowmap, internalEntries, gch);
    break;

  case COLORING_VB:
  case COLORING_VBBIT:
  case COLORING_VBCS:
    typedef typename Impl::GraphColor_VB <typename KernelHandle::GraphColoringHandleType, lno_row_view_t_, lno_nnz_view_t_> VBGraphColoring;
    gc = new VBGraphColoring(num_rows, internalRowmap, internalEntries, gch);
    break;

  case COLORING_VBD:
  case COLORING_VBDBIT:
    typedef typename Impl::GraphColor_VBD <typename KernelHandle::GraphColoringHandleType, lno_row_view_t_, lno_nnz_view_t_> VBDGraphColoring;
    gc = new VBDGraphColoring(num_rows, internalRowmap, internalEntries, gch);
    break;

  case COLORING_EB:
    typedef typename Impl::GraphColor_EB <typename KernelHandle::GraphColoringHandleType, lno_row_view_t_, lno_nnz_view_t_> EBGraphColoring;
    gc = new EBGraphColoring(num_rows, internalRowmap, internalEntries, gch);
    break;

  case COLORING_DEFAULT:
    break;

  default:
    break;
  }

  int num_phases = 0;
  if(sorted)
    gc->color_sorted_graph(colors_out, num_phases);
  else
    gc->color_graph(colors_out, num_phases);

  delete gc;
  double coloring_time = timer.seconds();
  gch->add_to_overall_coloring_time(coloring_time);
  gch->set_coloring_time(coloring_time);
  gch->set_num_phases(num_phases);
  gch->set_vertex_colors(colors_out);
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
///  Identical behavior to color_graph().
///  Deprecated because it uses the naming convention of sparse numerical algorithms
///  (separate symbolic and numeric phases) which doesn't apply to graph algorithms.
///  The old interface also took \c num_cols, which is not necessary as the adjacency
///  matrix must be square for coloring to make sense.
template <class KernelHandle,typename lno_row_view_t_, typename lno_nnz_view_t_>
void graph_color_symbolic(
    KernelHandle *handle,
    typename KernelHandle::nnz_lno_t num_verts,
    lno_row_view_t_ row_map,
    lno_nnz_view_t_ entries,
    bool is_symmetric = true)
{
  color_graph(handle, num_verts, num_verts, row_map, entries, is_symmetric, false);
}
#endif

}  // end namespace Experimental
}  // end namespace KokkosGraph

#endif   // _KOKKOSGRAPH_DISTANCE1_COLOR_HPP


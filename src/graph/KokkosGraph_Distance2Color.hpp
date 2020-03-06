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
// Questions? Contact William McLendon (wcmclen@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#ifndef _KOKKOS_GRAPH_COLORDISTANCE2_HPP
#define _KOKKOS_GRAPH_COLORDISTANCE2_HPP

#include "KokkosGraph_Distance1ColorHandle.hpp"
#include "KokkosGraph_Distance2ColorHandle.hpp"
#include "KokkosGraph_Distance1Color_impl.hpp" 
#include "KokkosGraph_Distance2Color_impl.hpp"

#include "KokkosKernels_Utils.hpp"


namespace KokkosGraph {

namespace Experimental {

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
/**
 * (DEPRECATED) Compute the left-side coloring of a bipartite matrix/graph.
 * Equivalent to color_bipartite_rows(), except this interface requires the user
 * to compute (col_map, col_entries) as the transpose of the graph (if nonsymmetric).
 *
 * This function is deprecated because it's not possible to support both undirected
 * distance-2 coloring and bipartite one-sided coloring
 * in a single interface. However, if the input graph has all diagonal entries present and
 * is symmetric (which is generally the case for discretized PDE matrices), then this
 * function is also equivalent to graph_color_distance2().
 *
 * In any case, the graphs (row_map, row_entries) and (col_map, col_entries) must be transposes
 * of each other.
 *
 * @param[in]  handle         The Kernel Handle
 * @param[in]  num_rows       Number of rows in the matrix (number of vertices)
 * @param[in]  num_cols       Number of columns in the matrix
 * @param[in]  row_map        Row map
 * @param[in]  row_entries    Row entries
 * @param[in]  col_map        Column map
 * @param[in]  col_entries    Column entries
 */
template<class KernelHandle, typename lno_row_view_t_, typename lno_nnz_view_t_, typename lno_col_view_t_, typename lno_colnnz_view_t_>
void graph_compute_distance2_color(KernelHandle *handle,
                                   typename KernelHandle::nnz_lno_t num_rows,
                                   typename KernelHandle::nnz_lno_t num_cols,
                                   lno_row_view_t_ row_map,
                                   lno_nnz_view_t_ row_entries,
                                   // If graph is symmetric, simply give same for col_map and row_map, and row_entries and col_entries.
                                   lno_col_view_t_ col_map,
                                   lno_colnnz_view_t_ col_entries)
{
    Kokkos::Impl::Timer timer;
    // Set our handle pointer to a GraphColoringHandleType.
    typename KernelHandle::GraphColorDistance2HandleType *gch_d2 = handle->get_distance2_graph_coloring_handle();
    // Get the algorithm we're running from the graph coloring handle.
    GraphColoringAlgorithmDistance2 algorithm = gch_d2->get_coloring_algo_type();
    // Create a view to save the colors to.
    using color_view_type = typename KernelHandle::GraphColorDistance2HandleType::color_view_type;
    color_view_type colors_out("Graph Colors", num_rows);

    Impl::GraphColorDistance2<typename KernelHandle::GraphColorDistance2HandleType, lno_row_view_t_, lno_nnz_view_t_, lno_col_view_t_, lno_colnnz_view_t_>
      gc(num_rows, num_cols, row_entries.extent(0), row_map, row_entries, col_map, col_entries, gch_d2);
    if(algorithm == COLORING_D2_SERIAL)
    {
      gc.compute_distance2_color_serial();
    }
    else if(algorithm == COLORING_D2_NB_BIT)
    {
      gc.compute_d2_coloring_dynamic();
    }
    else
    {
      gc.compute_distance2_color();
    }
    gch_d2->add_to_overall_coloring_time(timer.seconds());
    gch_d2->set_coloring_time(timer.seconds());
}
#endif

/**
 * Compute the distance-2 coloring of an undirected graph.
 *
 * The graph must be symmetric, but it is not required to have
 * diagonal entries. The coloring will not have distance-1 or distance-2
 * conflicts.
 *
 * @param[in]  handle         The Kernel Handle
 * @param[in]  num_vertices   Number of vertices in the graph
 * @param[in]  row_map        Row map
 * @param[in]  row_entries    Row entries
 *
 * \post <code>handle->get_distance2_graph_coloring_handle()->get_vertex_colors()</code>
 *    will return a view of length num_vertices, containing the colors.
 */

template<class KernelHandle, typename lno_row_view_t_, typename lno_nnz_view_t_>
void graph_color_distance2(
    KernelHandle *handle,
    typename KernelHandle::nnz_lno_t num_verts,
    lno_row_view_t_ row_map,
    lno_nnz_view_t_ row_entries);
{
    Kokkos::Impl::Timer timer;
    // Set our handle pointer to a GraphColoringHandleType.
    typename KernelHandle::GraphColorDistance2HandleType *gch_d2 = handle->get_distance2_graph_coloring_handle();
    // Get the algorithm we're running from the graph coloring handle.
    GraphColoringAlgorithmDistance2 algorithm = gch_d2->get_coloring_algo_type();
    // Create a view to save the colors to.
    using color_view_type = typename KernelHandle::GraphColorDistance2HandleType::color_view_type;
    color_view_type colors_out("Graph Colors", num_rows);
    Impl::GraphColorDistance2<typename KernelHandle::GraphColorDistance2HandleType, lno_row_view_t_, lno_nnz_view_t_, lno_col_view_t_, lno_colnnz_view_t_>
      gc(num_rows, num_rows, row_map, row_entries, row_map, row_entries, gch_d2);
    if(algorithm == COLORING_D2_SERIAL)
    {
      gc.compute_distance2_color_serial();
    }
    else if(algorithm == COLORING_D2_NB_BIT)
    {
      gc.compute_d2_coloring_dynamic();
    }
    else
    {
      gc.compute_distance2_color();
    }
    gch_d2->add_to_overall_coloring_time(timer.seconds());
    gch_d2->set_coloring_time(timer.seconds());
}

/**
 * Color the left part (rows) of a bipartite graph: rows r1 and r2 can have the same color 
 * if there is no column c such that edges (r1, c) and (r2, c) exist. This means only conflicts over a path
 * exactly two edges long are avoided.
 *
 * This problem is equivalent to grouping the matrix rows into a minimal number of sets, so that
 * within each set, the intersection of any two rows' entries is empty.
 *
 * Distance-1 conflicts (where r1 and c are neighbors) are not avoided,
 * since columns are not colored. In general, there is no particular relationship between a row and column
 * that happen to have the same index.
 *
 * However, if the input graph is symmetric and has diagonal entries in every row, then rows and columns
 * are equivalent and distance-1 conflicts are present through edges (r1, r1) and (r1, r2).
 *
 * @param[in]  handle         The Kernel Handle
 * @param[in]  num_rows       Number of "rows" (vertices in the left part of the graph)
 * @param[in]  num_columns    Number of "columns" (vertices in the right part of the graph)
 * @param[in]  row_map        Row map (CRS format)
 * @param[in]  row_entries    Row entries (CRS format)
 * @param[in]  is_symmetric   Whether (rowmap, row_entries) is known to be symmetric. If it
 *                            is, this saves computing the transpose internally.
 *
 * \post <code>handle->get_distance2_graph_coloring_handle()->get_vertex_colors()</code>
 *    will return a view of length num_rows, containing the colors.
 */

template<class KernelHandle, typename Rowmap, typename RowEntries>
void color_bipartite_rows(
    KernelHandle *handle,
    typename KernelHandle::nnz_lno_t num_rows,
    typename KernelHandle::nnz_lno_t num_columns,
    Rowmap rowmap,
    RowEntries row_entries,
    bool is_symmetric)
{
    Kokkos::Impl::Timer timer;
    // Set our handle pointer to a GraphColoringHandleType.
    typename KernelHandle::GraphColorDistance2HandleType *gch_d2 = handle->get_distance2_graph_coloring_handle();
    // Get the algorithm we're running from the graph coloring handle.
    GraphColoringAlgorithmDistance2 algorithm = gch_d2->get_coloring_algo_type();
    // Create a view to save the colors to.
    using color_view_type = typename KernelHandle::GraphColorDistance2HandleType::color_view_type;
    color_view_type colors_out("Graph Colors", num_rows);
    Impl::GraphColorDistance2<typename KernelHandle::GraphColorDistance2HandleType, lno_row_view_t_, lno_nnz_view_t_, lno_col_view_t_, lno_colnnz_view_t_>
      gc(num_rows, num_rows, row_map, row_entries, row_map, row_entries, gch_d2);
    if(algorithm == COLORING_D2_SERIAL)
    {
      gc.compute_distance2_color_serial();
    }
    else if(algorithm == COLORING_D2_NB_BIT)
    {
      gc.compute_d2_coloring_dynamic();
    }
    else
    {
      gc.compute_distance2_color();
    }
    gch_d2->add_to_overall_coloring_time(timer.seconds());
    gch_d2->set_coloring_time(timer.seconds());
}

/**
 * Color the right part (columns) of a bipartite graph: columns c1 and c2 can have the same color 
 * if there is no row r such that edges (r, c1) and (r, c2) exist.
 *
 * This problem is equivalent to grouping the matrix columns into a minimal number of sets, so that
 * within each set, no two columns appear together in any row's entries. This can be used for computing
 * a compressed Jacobian matrix.
 *
 * Note that the input to this function is still a CRS (row-wise) graph. If you have a CCS (column-wise)
 * or a symmetric graph, use color_bipartite_rows() instead. Calling that with the column-wise graph is
 * equivalent to calling this with the row-wise graph, and that way the transpose will be
 * computed automatically as needed.
 *
 * @param[in]  handle         The Kernel Handle
 * @param[in]  num_rows       Number of "rows" (vertices in the left part of the graph)
 * @param[in]  num_columns    Number of "columns" (vertices in the right part of the graph)
 * @param[in]  row_map        Row map
 * @param[in]  row_entries    Row entries
 *
 * \post handle->get_distance2_graph_coloring_handle()->get_vertex_colors() will return a view of length num_columns, containing the colors.
 */
template<class KernelHandle, typename Rowmap, typename RowEntries>
void color_bipartite_columns(
    KernelHandle *handle,
    typename KernelHandle::nnz_lno_t num_rows,
    typename KernelHandle::nnz_lno_t num_columns,
    Rowmap rowmap,
    RowEntries row_entries)
{
}

}      // end namespace Experimental
}      // end namespace KokkosGraph

#endif //_KOKKOS_GRAPH_COLORDISTANCE2_HPP


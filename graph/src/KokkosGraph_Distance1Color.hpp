//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#ifndef _KOKKOSGRAPH_DISTANCE1_COLOR_HPP
#define _KOKKOSGRAPH_DISTANCE1_COLOR_HPP

#include "KokkosGraph_color_d1_spec.hpp"
#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Utils.hpp"

namespace KokkosGraph {

namespace Experimental {

template <class KernelHandle, typename lno_row_view_t_, typename lno_nnz_view_t_>
void graph_color(const typename KernelHandle::HandleExecSpace& exec, *handle,
                 typename KernelHandle::nnz_lno_t num_vertices,
                 const lno_row_view_t_& row_map, const lno_nnz_view_t_& entries)
{
  typedef typename KernelHandle::HandleExecSpace ExecSpace;
  typedef typename KernelHandle::HandleTempMemorySpace MemSpace;
  typedef typename KernelHandle::HandlePersistentMemorySpace PersistentMemSpace;
  typedef typename Kokkos::Device<ExecSpace, MemSpace> DeviceType;

  typedef typename KernelHandle::const_size_type c_size_t;
  typedef typename KernelHandle::const_nnz_lno_t c_lno_t;
  typedef typename KernelHandle::const_nnz_scalar_t c_scalar_t;

  typedef typename KokkosKernels::Experimental::KokkosKernelsHandle<
      c_size_t, c_lno_t, c_scalar_t, ExecSpace, MemSpace, PersistentMemSpace>
      ConstKernelHandle;
  ConstKernelHandle tmp_handle(*handle);

  typedef Kokkos::View<typename lno_row_view_t_::const_value_type *,
                       typename KokkosKernels::Impl::GetUnifiedLayout<
                           lno_row_view_t_>::array_layout,
                       DeviceType, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      Internal_rowmap;
  typedef Kokkos::View<typename lno_nnz_view_t_::const_value_type *,
                       typename KokkosKernels::Impl::GetUnifiedLayout<
                           lno_nnz_view_t_>::array_layout,
                       DeviceType, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      Internal_entries;
  KokkosGraph::Impl::
      COLOR_D1<ConstKernelHandle, Internal_rowmap, Internal_entries>::color_d1(
          exec,
          &tmp_handle, num_vertices,
          Internal_rowmap(row_map.data(), row_map.extent(0)),
          Internal_entries(entries.data(), entries.extent(0)));
}

template <class KernelHandle, typename lno_row_view_t_, typename lno_nnz_view_t_>
void graph_color(KernelHandle *handle,
                 typename KernelHandle::nnz_lno_t num_rows,
                 const lno_row_view_t_& row_map, const lno_nnz_view_t_& entries)
{
  graph_color(typename KernelHandle::HandleExecSpace{}, handle, num_rows, row_map, entries);
}

template <class KernelHandle, typename lno_row_view_t_,
          typename lno_nnz_view_t_>
void graph_color [[deprecated("Please call graph_color([exec,] handle, num_rows, row_map, entries) instead"]]
    (KernelHandle *handle,
                 typename KernelHandle::nnz_lno_t num_rows,
                 typename KernelHandle::nnz_lno_t /* num_cols */,
                 lno_row_view_t_ row_map, lno_nnz_view_t_ entries,
                 bool is_symmetric = true) {
    if(!is_symmetric)
      throw(std::invalid_argument("graph_color: is_symmetric must be true and graph must be symmetric/undirected"));
    graph_color(handle, num_rows, row_map, entries);
}

template <class KernelHandle, typename lno_row_view_t_,
          typename lno_nnz_view_t_>
void graph_color_symbolic [[deprecated("Please call graph_color([exec,] handle, num_rows, row_map, entries) instead"]]
    (KernelHandle *handle,
                          typename KernelHandle::nnz_lno_t num_rows,
                          typename KernelHandle::nnz_lno_t /* num_cols */,
                          lno_row_view_t_ row_map, lno_nnz_view_t_ entries,
                          bool /* is_symmetric */ = true) {
    if(!is_symmetric)
      throw(std::invalid_argument("graph_color: is_symmetric must be true and graph must be symmetric/undirected"));
    graph_color(handle, num_rows, row_map, entries);
}

}  // end namespace Experimental
}  // end namespace KokkosGraph

#endif  // _KOKKOSGRAPH_DISTANCE1_COLOR_HPP

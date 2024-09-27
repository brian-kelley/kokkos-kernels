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

#include "KokkosSparse_rocsparse_tpl.hpp"
#include "KokkosSparse_Utils_rocsparse.hpp"

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE

namespace KokkosKernels {
namespace Impl {

TPLSingleton<rocsparseHandle_t>& TPLSingleton<rocsparse_handle>::getInstance()
{
  static TPLSingleton<rocsparse_handle> s;
  return s;
}

void TPLSingleton<rocsparse_handle>::initialize(rocsparse_handle& handle) {
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_create_handle(&handle));
}

void TPLSingleton<rocsparse_handle>::finalize(rocsparse_handle& handle) {
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_destroy_handle(handle));
}

}  // namespace Impl
}  // namespace KokkosKernels
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE


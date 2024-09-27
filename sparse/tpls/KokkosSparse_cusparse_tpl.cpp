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

#include "KokkosSparse_cusparse_tpl.hpp"

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include "KokkosSparse_Utils_cusparse.hpp"

namespace KokkosKernels {
namespace Impl {

TPLSingleton<cusparseHandle_t>& TPLSingleton<cusparseHandle_t>::getInstance()
{
  static TPLSingleton<cusparseHandle_t> s;
  return s;
}

void TPLSingleton<cusparseHandle_t>::initialize(cusparseHandle_t& handle) {
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreate(&handle));
}

void TPLSingleton<cusparseHandle_t>::finalize(cusparseHandle_t& handle) {
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroy(handle));
}

}  // namespace Impl
}  // namespace KokkosKernels
#endif // KOKKOSKERNELS_ENABLE_TPL_CUSPARSE


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

#ifndef KOKKOSBLAS_ROCBLAS_TPL_HPP
#define KOKKOSBLAS_ROCBLAS_TPL_HPP

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#include <rocblas/rocblas.h>

namespace KokkosBlas::Impl {

using RocBLASSingleton = ::KokkosKernels::Impl::TPLSingleton<rocblas_handle>;

void rocblas_internal_error_throw(rocblas_status rocblasState, const char* name, const char* file,
                                         const int line);

void rocblas_internal_safe_call(rocblas_status rocblasState, const char* name, const char* file = nullptr,
                                       const int line = 0);

// The macro below defines the interface for the safe rocblas calls.
// The functions themselves are protected by impl namespace and this
// is not meant to be used by external application or libraries.
#define KOKKOS_ROCBLAS_SAFE_CALL_IMPL(call) \
  KokkosBlas::Impl::rocblas_internal_safe_call(call, #call, __FILE__, __LINE__)

/// \brief This function converts KK transpose mode to rocBLAS transpose mode
inline rocblas_operation trans_mode_kk_to_rocblas(const char kkMode[]) {
  rocblas_operation trans;
  if ((kkMode[0] == 'N') || (kkMode[0] == 'n'))
    trans = rocblas_operation_none;
  else if ((kkMode[0] == 'T') || (kkMode[0] == 't'))
    trans = rocblas_operation_transpose;
  else
    trans = rocblas_operation_conjugate_transpose;
  return trans;
}

}  // namespace KokkosBlas::Impl

#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#endif


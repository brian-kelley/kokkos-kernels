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

#ifndef KOKKOSBLAS_CUBLAS_TPL_HPP
#define KOKKOSBLAS_CUBLAS_TPL_HPP

#include <KokkosKernels_config.h>
#include "KokkosKernels_TPLSingleton.hpp"

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include "cuda_runtime.h"
#include "cublas_v2.h"

namespace KokkosBlas::Impl {

using cuBLASSingleton = ::KokkosKernels::Impl::TPLSingleton<cublasHandle_t>;

void cublas_internal_error_throw(cublasStatus_t cublasState, const char* name, const char* file,
                                        const int line);

void cublas_internal_safe_call(cublasStatus_t cublasState, const char* name, const char* file = nullptr,
                                      const int line = 0);

// The macro below defines the interface for the safe cublas calls.
// The functions themselves are protected by impl namespace and this
// is not meant to be used by external application or libraries.
#define KOKKOS_CUBLAS_SAFE_CALL_IMPL(call) KokkosBlas::Impl::cublas_internal_safe_call(call, #call, __FILE__, __LINE__)

/// \brief This function converts KK transpose mode to cuBLAS transpose mode
inline cublasOperation_t trans_mode_kk_to_cublas(const char kkMode[]) {
  cublasOperation_t trans;
  if ((kkMode[0] == 'N') || (kkMode[0] == 'n'))
    trans = CUBLAS_OP_N;
  else if ((kkMode[0] == 'T') || (kkMode[0] == 't'))
    trans = CUBLAS_OP_T;
  else
    trans = CUBLAS_OP_C;
  return trans;
}

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#endif


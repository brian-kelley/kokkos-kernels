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

#include <Kokkos_Core.hpp>

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include "KokkosBlas_cublas_tpl.hpp"

namespace KokkosBlas::Impl {

void cublas_internal_error_throw(cublasStatus_t cublasState, const char* name, const char* file,
                                        const int line) {
  std::ostringstream out;
  // out << name << " error( " << cublasGetStatusName(cublasState)
  //     << "): " << cublasGetStatusString(cublasState);
  out << name << " error( ";
  switch (cublasState) {
    case CUBLAS_STATUS_NOT_INITIALIZED:
      out << "CUBLAS_STATUS_NOT_INITIALIZED): the library was not initialized.";
      break;
    case CUBLAS_STATUS_ALLOC_FAILED: out << "CUBLAS_STATUS_ALLOC_FAILED): the resource allocation failed."; break;
    case CUBLAS_STATUS_INVALID_VALUE:
      out << "CUBLAS_STATUS_INVALID_VALUE): an invalid numerical value was "
             "used as an argument.";
      break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
      out << "CUBLAS_STATUS_ARCH_MISMATCH): an absent device architectural "
             "feature is required.";
      break;
    case CUBLAS_STATUS_MAPPING_ERROR:
      out << "CUBLAS_STATUS_MAPPING_ERROR): an access to GPU memory space "
             "failed.";
      break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
      out << "CUBLAS_STATUS_EXECUTION_FAILED): the GPU program failed to "
             "execute.";
      break;
    case CUBLAS_STATUS_INTERNAL_ERROR: out << "CUBLAS_STATUS_INTERNAL_ERROR): an internal operation failed."; break;
    case CUBLAS_STATUS_NOT_SUPPORTED:
      out << "CUBLAS_STATUS_NOT_SUPPORTED): the feature required is not "
             "supported.";
      break;
    default: out << "unrecognized error code): this is bad!"; break;
  }
  if (file) {
    out << " " << file << ":" << line;
  }
  throw std::runtime_error(out.str());
}

void cublas_internal_safe_call(cublasStatus_t cublasState, const char* name, const char* file = nullptr,
                                      const int line = 0) {
  if (CUBLAS_STATUS_SUCCESS != cublasState) {
    cublas_internal_error_throw(cublasState, name, file, line);
  }
}

} // KokkosBlas::Impl

namespace KokkosKernels::Impl {

TPLSingleton<cublasHandle_t>& TPLSingleton<cublasHandle_t>::getInstance()
{
  static TPLSingleton<cublasHandle_t> s;
  return s;
}

void TPLSingleton<cublasHandle_t>::initialize(cublasHandle_t& handle) {
  KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasCreate(&handle));
}

void TPLSingleton<cublasHandle_t>::finalize(cublasHandle_t& handle) {
  KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasDestroy(handle));
}

} // KokkosKernels::Impl

#endif


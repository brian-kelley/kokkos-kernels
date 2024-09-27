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

#include "KokkosLapack_cusolver_tpl.hpp"

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSOLVER

namespace KokkosKernels::Impl {

TPLSingleton<cusolverDnHandle_t>& TPLSingleton<cusolverDnHandle_t>::getInstance()
{
  static TPLSingleton<cusolverDnHandle_t> s;
  return s;
}

void TPLSingleton<cusolverDnHandle_t>::initialize(cusolverDnHandle_t& handle) {
  cusolverStatus_t stat = cusolverDnCreate(&handle);
  if (stat != CUSOLVER_STATUS_SUCCESS) Kokkos::abort("CUSOLVER initialization failed\n");
}

void TPLSingleton<cusolverDnHandle_t>::finalize(cusolverDnHandle_t& handle) {
  cusolverDnDestroy(handle);
}

} // KokkosKernels::Impl

namespace KokkosLapack::Impl {

void cusolver_internal_error_throw(cusolverStatus_t cusolverStatus, const char* name, const char* file,
                                          const int line) {
  std::ostringstream out;
  out << name << " error( ";
  switch (cusolverStatus) {
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      out << "CUSOLVER_STATUS_NOT_INITIALIZED): cusolver handle was not "
             "created correctly.";
      break;
    case CUSOLVER_STATUS_ALLOC_FAILED:
      out << "CUSOLVER_STATUS_ALLOC_FAILED): you might tried to allocate too "
             "much memory";
      break;
    case CUSOLVER_STATUS_INVALID_VALUE: out << "CUSOLVER_STATUS_INVALID_VALUE)"; break;
    case CUSOLVER_STATUS_ARCH_MISMATCH: out << "CUSOLVER_STATUS_ARCH_MISMATCH)"; break;
    case CUSOLVER_STATUS_EXECUTION_FAILED: out << "CUSOLVER_STATUS_EXECUTION_FAILED)"; break;
    case CUSOLVER_STATUS_INTERNAL_ERROR: out << "CUSOLVER_STATUS_INTERNAL_ERROR)"; break;
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: out << "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED)"; break;
    default: out << "unrecognized error code): this is bad!"; break;
  }
  if (file) {
    out << " " << file << ":" << line;
  }
  throw std::runtime_error(out.str());
}

void cusolver_internal_safe_call(cusolverStatus_t cusolverStatus, const char* name, const char* file = nullptr,
                                        const int line = 0) {
  if (CUSOLVER_STATUS_SUCCESS != cusolverStatus) {
    cusolver_internal_error_throw(cusolverStatus, name, file, line);
  }
}

} // KokkosLapack::Impl

#endif // KOKKOSKERNELS_ENABLE_TPL_CUSOLVER

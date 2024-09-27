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
#include <KokkosKernels_config.h>
#include "KokkosKernels_TPLSingleton.hpp"
#include "KokkosBlas_rocblas_tpl.hpp"

namespace KokkosBlas::Impl {

void rocblas_internal_error_throw(rocblas_status rocblasState, const char* name, const char* file,
                                         const int line) {
  std::ostringstream out;
  out << name << " error( ";
  switch (rocblasState) {
    case rocblas_status_invalid_handle:
      out << "rocblas_status_invalid_handle): handle not initialized, invalid "
             "or null.";
      break;
    case rocblas_status_not_implemented: out << "rocblas_status_not_implemented): function is not implemented."; break;
    case rocblas_status_invalid_pointer: out << "rocblas_status_invalid_pointer): invalid pointer argument."; break;
    case rocblas_status_invalid_size: out << "rocblas_status_invalid_size): invalid size argument."; break;
    case rocblas_status_memory_error:
      out << "rocblas_status_memory_error): failed internal memory allocation, "
             "copy or dealloc.";
      break;
    case rocblas_status_internal_error: out << "rocblas_status_internal_error): other internal library failure."; break;
    case rocblas_status_perf_degraded:
      out << "rocblas_status_perf_degraded): performance degraded due to low "
             "device memory.";
      break;
    case rocblas_status_size_query_mismatch: out << "unmatched start/stop size query): ."; break;
    case rocblas_status_size_increased:
      out << "rocblas_status_size_increased): queried device memory size "
             "increased.";
      break;
    case rocblas_status_size_unchanged:
      out << "rocblas_status_size_unchanged): queried device memory size "
             "unchanged.";
      break;
    case rocblas_status_invalid_value: out << "rocblas_status_invalid_value): passed argument not valid."; break;
    case rocblas_status_continue:
      out << "rocblas_status_continue): nothing preventing function to "
             "proceed.";
      break;
    case rocblas_status_check_numerics_fail:
      out << "rocblas_status_check_numerics_fail): will be set if the "
             "vector/matrix has a NaN or an Infinity.";
      break;
    default: out << "unrecognized error code): this is bad!"; break;
  }
  if (file) {
    out << " " << file << ":" << line;
  }
  throw std::runtime_error(out.str());
}

void rocblas_internal_safe_call(rocblas_status rocblasState, const char* name, const char* file = nullptr,
                                       const int line = 0) {
  if (rocblas_status_success != rocblasState) {
    rocblas_internal_error_throw(rocblasState, name, file, line);
  }
}
}

namespace KokkosKernels::Impl {

TPLSingleton<rocblas_handle>& TPLSingleton<rocblas_handle>::getInstance()
{
  static TPLSingleton<rocblas_handle> s;
  return s;
}

void TPLSingleton<rocblas_handle>::initialize(rocblas_handle& handle) {
  KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_create_handle(&handle))
}

void TPLSingleton<rocblas_handle>::finalize(rocblas_handle& handle) {
  KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_destroy_handle(handle));
}

}


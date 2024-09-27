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

#ifndef KOKKOSLAPACK_CUSOLVER_HPP_
#define KOKKOSLAPACK_CUSOLVER_HPP_

#include <KokkosKernels_config.h>

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSOLVER
#include <cusolverDn.h>

namespace KokkosLapack::Impl {
  using 
}


void cusolver_internal_error_throw(cusolverStatus_t cusolverStatus, const char* name, const char* file,
                                          const int line);
void cusolver_internal_safe_call(cusolverStatus_t cusolverStatus, const char* name, const char* file = nullptr,
                                        const int line = 0);

// The macro below defines is the public interface for the safe cusolver calls.
// The functions themselves are protected by impl namespace.
#define KOKKOS_CUSOLVER_SAFE_CALL_IMPL(call) \
  KokkosLapack::Impl::cusolver_internal_safe_call(call, #call, __FILE__, __LINE__)

}  // namespace Impl
}  // namespace KokkosLapack

#endif  // KOKKOSKERNELS_ENABLE_TPL_CUSOLVER
#endif  // KOKKOSLAPACK_CUSOLVER_HPP_

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

#include "KokkosKernels_EagerInitialize.hpp"
#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"

// Include the minimal set of headers that declare all TPL singletons.
// These are safe to include whether or not the TPL is enabled.
#ifdef KOKKOSKERNELS_ENABLE_COMPONENT_BLAS
#include "KokkosBlas_cublas_tpl.hpp"
#include "KokkosBlas_rocblas_tpl.hpp"
#endif

#ifdef KOKKOSKERNELS_ENABLE_COMPONENT_SPARSE
#include "KokkosSparse_cusparse_tpl.hpp"
#include "KokkosSparse_rocsparse_tpl.hpp"
#endif

#ifdef KOKKOSKERNELS_ENABLE_COMPONENT_LAPACK
#include "KokkosLapack_cusolver_tpl.hpp"
#endif

#include "KokkosKernels_magma_tpl.hpp"

namespace KokkosKernels {
void eager_initialize() {
  if (!Kokkos::is_initialized()) {
    throw std::runtime_error("Kokkos::intialize must be called before KokkosKernels::eager_initialize");
  }
#ifdef KOKKOSKERNELS_ENABLE_COMPONENT_BLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
  (void)KokkosBlas::Impl::cublasSingleton::singleton();
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
  (void)KokkosBlas::Impl::rocblasSingleton::singleton();
#endif
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
  (void)KokkosKernels::Impl::magmaSingleton::singleton();
#endif

#ifdef KOKKOSKERNELS_ENABLE_COMPONENT_SPARSE
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
  (void)KokkosKernels::Impl::cusparseSingleton::singleton();
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
  (void)KokkosKernels::Impl::rocsparseSingleton::singleton();
#endif
#endif

#ifdef KOKKOSKERNELS_ENABLE_COMPONENT_LAPACK
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSOLVER
  (void)KokkosLapack::Impl::cusolverSingleton::singleton();
#endif
#endif
}
}  // namespace KokkosKernels

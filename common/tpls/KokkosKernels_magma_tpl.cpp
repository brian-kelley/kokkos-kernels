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

#include <KokkosKernels_config.h>
#include "KokkosKernels_TPLSingleton.hpp"
#include "KokkosKernels_magma_tpl.hpp"

#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include "magma_v2.h"

namespace KokkosKernels::Impl {

TPLSingleton<MagmaDummyHandle>& TPLSingleton<MagmaDummyHandle>::getInstance()
{
  static TPLSingleton<MagmaDummyHandle> s;
  return s;
}

void TPLSingleton<MagmaDummyHandle>::initialize(MagmaDummyHandle&) {
  magma_int_t stat = magma_init();
  if (stat != MAGMA_SUCCESS) Kokkos::abort("MAGMA initialization failed\n");
}

void TPLSingleton<MagmaDummyHandle>::finalize(MagmaDummyHandle&) {
  magma_finalize();
}

} // KokkosKernels::Impl

#endif // KOKKOSKERNELS_ENABLE_TPL_MAGMA


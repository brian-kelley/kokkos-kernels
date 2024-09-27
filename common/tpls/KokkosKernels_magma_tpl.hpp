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
#ifndef KOKKOSBLAS_MAGMA_TPL_HPP_
#define KOKKOSBLAS_MAGMA_TPL_HPP_

#include "KokkosKernels_TPLSingleton.hpp"

// Magma is used by both BLAS and LAPACK components (and neither depends on the other),
// so put its initialize/finalize logic here in Common.
namespace KokkosKernels::Impl {
  // Magma doesn't use a handle type.
  // Use this dummy type as the TPLSingleton template parameter.
  struct MagmaDummyHandle {};

  using magmaSingleton = ::KokkosKernels::Impl::TPLSingleton<MagmaDummyHandle>;
}

#endif


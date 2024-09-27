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

#ifndef KOKKOSSPARSE_ROCSPARSE_TPL_HPP
#define KOKKOSSPARSE_ROCSPARSE_TPL_HPP

#include <KokkosKernels_config.h>
#include "KokkosKernels_TPLSingleton.hpp"

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#include <rocsparse/rocsparse.h>

namespace KokkosSparse::Impl {
  using rocsparseSingleton = ::KokkosKernels::Impl::TPLSingleton<rocsparse_handle>;

}

#endif // KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#endif // KOKKOSSPARSE_ROCSPARSE_TPL_HPP


// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSKERNELS_DEFAULT_TYPES_H
#define KOKKOSKERNELS_DEFAULT_TYPES_H

#include "Kokkos_Core.hpp"         //for LayoutLeft/LayoutRight
#include <KokkosKernels_config.h>  //for all the ETI #cmakedefine macros

#define KK_IMPL_MAKE_TYPE_ALIAS(symbol, type) \
  namespace KokkosKernels {                   \
  using symbol = type;                        \
  }

#if defined(KOKKOSKERNELS_INST_ORDINAL_INT)
KK_IMPL_MAKE_TYPE_ALIAS(default_lno_t, int)
#elif defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T)
KK_IMPL_MAKE_TYPE_ALIAS(default_lno_t, int64_t)
#else
// Non-ETI build: default to int
KK_IMPL_MAKE_TYPE_ALIAS(default_lno_t, int)
#endif
// Prefer int as the default offset type, because cuSPARSE doesn't support
// size_t for rowptrs.
#if defined(KOKKOSKERNELS_INST_OFFSET_INT)
KK_IMPL_MAKE_TYPE_ALIAS(default_size_type, int)
#elif defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)
KK_IMPL_MAKE_TYPE_ALIAS(default_size_type, size_t)
#else
// Non-ETI build: default to int
KK_IMPL_MAKE_TYPE_ALIAS(default_size_type, int)
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
KK_IMPL_MAKE_TYPE_ALIAS(default_layout, Kokkos::LayoutLeft)
#elif defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
KK_IMPL_MAKE_TYPE_ALIAS(default_layout, Kokkos::LayoutRight)
#else
KK_IMPL_MAKE_TYPE_ALIAS(default_layout, Kokkos::LayoutLeft)
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
KK_IMPL_MAKE_TYPE_ALIAS(default_scalar, double)
#elif defined(KOKKOSKERNELS_INST_FLOAT)
KK_IMPL_MAKE_TYPE_ALIAS(default_scalar, float)
#elif defined(KOKKOSKERNELS_INST_HALF)
KK_IMPL_MAKE_TYPE_ALIAS(default_scalar, Kokkos::Experimental::half_t)
#elif defined(KOKKOSKERNELS_INST_BHALF)
KK_IMPL_MAKE_TYPE_ALIAS(default_scalar, Kokkos::Experimental::bhalf_t)
#else
KK_IMPL_MAKE_TYPE_ALIAS(default_scalar, double)
#endif

#if defined(KOKKOS_ENABLE_CUDA)
KK_IMPL_MAKE_TYPE_ALIAS(default_device, Kokkos::Cuda)
#elif defined(KOKKOS_ENABLE_HIP)
KK_IMPL_MAKE_TYPE_ALIAS(default_device, Kokkos::HIP)
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
KK_IMPL_MAKE_TYPE_ALIAS(default_device, Kokkos::Experimental::OpenMPTarget)
#elif defined(KOKKOS_ENABLE_OPENMP)
KK_IMPL_MAKE_TYPE_ALIAS(default_device, Kokkos::OpenMP)
#elif defined(KOKKOS_ENABLE_THREADS)
KK_IMPL_MAKE_TYPE_ALIAS(default_device, Kokkos::Threads)
#else
KK_IMPL_MAKE_TYPE_ALIAS(default_device, Kokkos::Serial)
#endif

namespace KokkosKernels {
template <typename exec_space>
struct default_memspace {
  static_assert(Kokkos::is_execution_space_v<exec_space>,
                "default_memspace<T> requires that T is an execution space type.");
  using type = typename exec_space::memory_space;
};

#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct default_memspace<Kokkos::Cuda> {
#if defined(KOKKOSKERNELS_INST_MEMSPACE_CUDAUVMSPACE) && !defined(KOKKOSKERNELS_INST_MEMSPACE_CUDASPACE)
  using type = Kokkos::CudaUVMSpace;
#else
  using type = Kokkos::CudaSpace;
#endif
};
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <>
struct default_memspace<Kokkos::HIP> {
#if defined(KOKKOSKERNELS_INST_MEMSPACE_HIPMANAGEDSPACE) && !defined(KOKKOSKERNELS_INST_MEMSPACE_HIPSPACE)
  using type = Kokkos::HIPManagedSpace;
#else
  using type = Kokkos::HIPSpace;
#endif
};
#endif

#if defined(KOKKOS_ENABLE_SYCL)
template <>
struct default_memspace<Kokkos::SYCL> {
#if defined(KOKKOSKERNELS_INST_MEMSPACE_SYCLSHAREDSPACE) && !defined(KOKKOSKERNELS_INST_MEMSPACE_SYCLSPACE)
  using type = Kokkos::SYCLSharedUSMSpace;
#else
  using type = Kokkos::SYCLDeviceUSMSpace;
#endif
};
#endif

template <typename exec_space>
using default_memspace_t = typename default_memspace<exec_space>::type;
}  // namespace KokkosKernels

#undef KK_IMPL_MAKE_TYPE_ALIAS

#endif  // KOKKOSKERNELS_DEFAULT_TYPES_H

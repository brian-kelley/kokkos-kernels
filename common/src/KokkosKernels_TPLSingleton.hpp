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

#ifndef KOKKOSKERNELS_TPLSINGLETON_HPP
#define KOKKOSKERNELS_TPLSINGLETON_HPP

#include "Kokkos_Core.hpp"
#include <memory>

namespace KokkosKernels {
namespace Impl {

  template<typename T>
  struct TPLSingleton
  {
    static T& singleton() {
      TPLSingleton<T>& instance = getInstance();
      if(!instance.data) {
        instance.data = std::make_unique<T>();
        initialize(*instance.data());
        Kokkos::push_finalize_hook(
            [&]() {
              finalize(*instance.data());
              instance.data().reset();
            });
      }
      return *data;
    }

    static bool isInitialized() {
      // The underlying object is initialized
      // if and only if data is non-null.
      return getInstance().data;
    }

private:
    void initialize(T&);

    void finalize(T&);

    static TPLSingleton<T>& getInstance();

    std::unique_ptr<T> data;
  };

}
}

#endif


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

#ifndef KOKKOSKERNELS_SEGMENTED_REDUCESCAN_HPP
#define KOKKOSKERNELS_SEGMENTED_REDUCESCAN_HPP

#include <Kokkos_Core.hpp>

namespace KokkosKernels {

template<typename Value>
KOKKOS_INLINE_FUNCTION
void updateSegmentedReduce(Kokkos::pair<Value, bool>& update, Value val, bool beginsSegment)
{
  if(!src.second) {
    // src flag is set, so update dest's value
    dest.first += src.first;
  }
  // and update dest's flag
  dest.second = dest.second || src.second;
}

template <class Value>
struct SegmentedScan {
 public:
  // Required
  using reducer    = SegmentedPrefixSum<Value>;
  using value_type = Kokkos::pair<Value, bool>;
  using result_view_type = Kokkos::View<value_type, Kokkos::AnonymousSpace>;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  SegmentedScan(value_type& value_)
  : value(&value_), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  SegmentedScan(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const
  {
    if(!src.second) {
      // src flag is set, so update dest's value
      dest.first += src.first;
    }
    // and update dest's flag
    dest.second = dest.second || src.second;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.first = Kokkos::reduction_identity<Value>::sum();
    val.second = false;
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

} // namespace KokkosKernels

#endif


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

#include "KokkosKernels_Segmented_ReduceScan.hpp"
#include <KokkosKernels_IOUtils.hpp>

template<typename HostValueView, typename HostSegmentView, typename Device>
void test_segmented_reduce_impl(const HostValueView& valuesHost, const HostSegmentView& segmentsHost, int nsegments)
{
  using Value = typename HostValueView::value_type;
  using ExecSpace = typename Device::execution_space;
  using MemSpace = typename Device::memory_space;
  using Reducer = KokkosKernels::SegmentedScan<Value, Kokkos::HostSpace>;
  using ReducerValue = typename Reducer::value_type;
  auto values = Kokkos::create_mirror(MemSpace(), valuesHost);
  auto segments = Kokkos::create_mirror(MemSpace(), segmentsHost);
  Kokkos::deep_copy(values, valuesHost);
  Kokkos::deep_copy(segments, segmentsHost);
  decltype(values) reductions("reductionsActual", nsegments);
  Kokkos::parallel_scan(Kokkos::RangePolicy<ExecSpace>(0, values.extent(0)),
      KOKKOS_LAMBDA(int i, ReducerValue& update, bool finalPass)
      {
        // Segmented reduce is just an inclusive segmented scan,
        // Is element i the first one in its segment?
        // Note: this doesn't need to include i==0, because there are no values before that to accumulate.
        bool startsSegment = (i > 0) && segments(i) != segments(i-1);
      });
  HostValueView reductionsGold("reductionsGold", nsegments);
  HostValueView reductionsActual("reductionsActual", nsegments);
  Kokkos::deep_copy(reductionsActual, reductions

  // Compute the correct sum for each segment
  HostValueView reductionsGold("valuesGold", nsegments);
  for(size_t i = 0; i < valuesHost.extent(0); i++) {
    reductionsGold(segmentsHost(i)) += valuesHost(i);
  }
}

template<typename HostValueView, typename HostSegmentView>
void generateValuesAndSegments(int n, int numSegments, HostValueView& values, HostSegmentView& segments)
{
  using Value = typename HostValueView::value_type;
  values = HostValueView("values", n);
  segments = HostSegmentView("segments", n);
  // Populate randomized values
  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(13718);
  KokkosKerngetRandomBounds
  Kokkos::fill_random(values, rand_pool, randomUpperBound<Value>(10.0));
}

template<typename ValueView, typename SegmentView, typename Device>
void test_segmented_reduce()
{

}

template<typename ValueView, typename SegmentView, typename Device>
void test_segmented_scan_impl(const ValueView& values, const SegmentView& segments, int nsegments, bool inclusive)
{
  // Test with varying numbers of segments
}

template<typename Value, typename Segment, typename Device>
void test_segmented_reduce(int n, int numSegments)
{
  // Generate test input
}

TEST_F(TestCategory, common_segmented_reduce_##TestDevice)
{
  test_segmented_reduce<float, int, TestDevice>();
  test_segmented_reduce<float, size_t, TestDevice>();
  test_segmented_reduce<double, int, TestDevice>();
  test_segmented_reduce<int, size_t, TestDevice>();
  test_segmented_reduce<Kokkos::complex<double>, int, TestDevice>();
}

TEST_F(TestCategory, common_segmented_inclusive_scan_##TestDevice)
{
  test_segmented_scan<float, int, TestDevice>(true);
  test_segmented_scan<float, size_t, TestDevice>(true);
  test_segmented_scan<double, int, TestDevice>(true);
  test_segmented_scan<int, size_t, TestDevice>(true);
  test_segmented_scan<Kokkos::complex<double>, int, TestDevice>(true);
}

TEST_F(TestCategory, common_segmented_exclusive_scan_##TestDevice)
{
  test_segmented_scan<float, int, TestDevice>(false);
  test_segmented_scan<float, size_t, TestDevice>(false);
  test_segmented_scan<double, int, TestDevice>(false);
  test_segmented_scan<int, size_t, TestDevice>(false);
  test_segmented_scan<Kokkos::complex<double>, int, TestDevice>(false);
}


// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_IMPL_GETRS_HPP_
#define KOKKOSLAPACK_IMPL_GETRS_HPP_

/// \file KokkosLapack_getrs_impl.hpp
/// \brief Implementation(s) of solve using LU factors.

#include <KokkosKernels_config.h>
#include <KokkosKernels_ArithTraits.hpp>
#include <KokkosBlas3_trsm.hpp>

namespace KokkosLapack {
namespace Impl {

template <class IpivView, class BMatrix>
struct laswp_functor {
  IpivView m_Ipiv;
  BMatrix m_B;
  bool m_forward;

  laswp_functor(const IpivView& Ipiv, const BMatrix& B, bool forward)
      : m_Ipiv(Ipiv), m_B(B), m_forward(forward) {}

  void KOKKOS_FUNCTION operator()(const int colIdx) const {
    typename BMatrix::non_const_value_type tmp;
    const int npiv = m_Ipiv.extent_int(0);
    if (m_forward) {
      // Forward: apply pivots from first to last (used for trans='N')
      for (int rowIdx = 0; rowIdx < npiv; ++rowIdx) {
        const int piv       = m_Ipiv(rowIdx) - 1;  // Convert from 1-based to 0-based
        tmp                 = m_B(rowIdx, colIdx);
        m_B(rowIdx, colIdx) = m_B(piv, colIdx);
        m_B(piv, colIdx)    = tmp;
      }
    } else {
      // Backward: apply pivots in reverse order (used for trans='T'/'C', applies P^T)
      for (int rowIdx = npiv - 1; rowIdx >= 0; --rowIdx) {
        const int piv       = m_Ipiv(rowIdx) - 1;  // Convert from 1-based to 0-based
        tmp                 = m_B(rowIdx, colIdx);
        m_B(rowIdx, colIdx) = m_B(piv, colIdx);
        m_B(piv, colIdx)    = tmp;
      }
    }
  }
};

template <class ExecutionSpace, class AMatrix, class IpivView, class BMatrix, class InfoView>
void getrs_impl(const ExecutionSpace& space, const char trans[], const AMatrix& A, const IpivView& Ipiv,
                const BMatrix& B, const InfoView& /* Info */) {
  auto one = KokkosKernels::ArithTraits<typename AMatrix::non_const_value_type>::one();

  if (trans[0] == 'N' || trans[0] == 'n') {
    Kokkos::parallel_for(Kokkos::RangePolicy(space, 0, B.extent(1)), laswp_functor(Ipiv, B, true));
    KokkosBlas::trsm(space, "L", "L", "N", "U", one, A, B);
    KokkosBlas::trsm(space, "L", "U", "N", "N", one, A, B);
  } else {
    KokkosBlas::trsm(space, "L", "U", trans, "N", one, A, B);
    KokkosBlas::trsm(space, "L", "L", trans, "U", one, A, B);
    Kokkos::parallel_for(Kokkos::RangePolicy(space, 0, B.extent(1)), laswp_functor(Ipiv, B, false));
  }
}

}  // namespace Impl
}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_IMPL_GETRS_HPP_

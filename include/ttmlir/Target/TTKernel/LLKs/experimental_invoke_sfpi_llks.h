// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_INVOKE_SFPI_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_INVOKE_SFPI_LLKS_H

#include "cmath_common.h"

namespace experimental {

template <typename Callable, typename... Args>
inline void _llk_math_user_sfpi_(Callable &&sfpu_func, Args &&...args) {
  math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(0);
  math::set_addr_mod_base();

  TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

  std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);

  math::clear_dst_reg_addr();

  TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
  math::clear_addr_mod_base();
}

template <typename Callable, typename... Args>
inline void invoke_sfpi(Callable &&sfpu_func, Args &&...args) {
  MATH((_llk_math_user_sfpi_(std::forward<Callable>(sfpu_func),
                             std::forward<Args>(args)...)));
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_INVOKE_SFPI_LLKS_H

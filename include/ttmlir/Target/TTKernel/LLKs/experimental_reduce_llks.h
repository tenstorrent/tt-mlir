// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REDUCE_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REDUCE_LLKS_H

namespace experimental {
using std::uint32_t;

template <PoolType reduce_type, ReduceDim reduce_dim>
ALWI void reduce_init_short(uint32_t icb, uint32_t icb_scaler, uint32_t ocb,
                            bool disbale_mask = false) {
  UNPACK((llk_unpack_AB_reduce_init<reduce_dim>(icb, icb_scaler)));
  MATH((llk_math_reduce_init<reduce_type, reduce_dim, MATH_FIDELITY>()));
  if (disbale_mask) {
    PACK((llk_pack_reduce_config_v2<reduce_dim, false, true, DST_ACCUM_MODE>(
        ocb)));
  } else {
    PACK((llk_pack_reduce_config_v2<reduce_dim, false, false, DST_ACCUM_MODE>(
        ocb)));
  }
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REDUCE_LLKS_H

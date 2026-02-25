// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_UNTILIZE_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_UNTILIZE_LLKS_H

namespace experimental {
using std::uint32_t;

// Untilize implementation using pack_untilize (packer-side layout
// transformation). This replaces the deprecated unpack_untilize approach and
// properly supports UnpackToDestEn for fp32 precision preservation.
template <uint32_t block_ct_dim>
ALWI void untilize_block_impl(uint32_t icb, uint32_t ocb, uint32_t block_r) {
  // Initialize for pack_untilize mode. This overrides the preceding
  // untilize_init() which configured the deprecated unpack_untilize path.
  ckernel::pack_untilize_init<block_ct_dim>(icb, ocb);

  for (uint32_t r = 0; r < block_r; ++r) {
    MATH((llk_math_wait_for_dest_available()));
    for (uint32_t c = 0; c < block_ct_dim; ++c) {
      uint32_t tile_idx = r * block_ct_dim + c;
      UNPACK((llk_unpack_A<BroadcastType::NONE, false,
                           EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
          icb, tile_idx)));
      MATH((llk_math_eltwise_unary_datacopy<
            A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(c)));
    }
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));

    PACK((llk_packer_wait_for_math_done()));
    PACK((llk_pack_untilize<block_ct_dim>(1, ocb)));
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
  }

  ckernel::pack_untilize_uninit(ocb);
}

ALWI void untilize_block(uint32_t icb, uint32_t ocb, uint32_t block_r,
                         uint32_t block_c) {
  // Dispatch to templated implementation. block_ct_dim must be a compile-time
  // constant for the pack_untilize MOP configuration.
  switch (block_c) {
  case 1:
    untilize_block_impl<1>(icb, ocb, block_r);
    break;
  case 2:
    untilize_block_impl<2>(icb, ocb, block_r);
    break;
  case 3:
    untilize_block_impl<3>(icb, ocb, block_r);
    break;
  case 4:
    untilize_block_impl<4>(icb, ocb, block_r);
    break;
  case 5:
    untilize_block_impl<5>(icb, ocb, block_r);
    break;
  case 6:
    untilize_block_impl<6>(icb, ocb, block_r);
    break;
  case 7:
    untilize_block_impl<7>(icb, ocb, block_r);
    break;
  case 8:
    untilize_block_impl<8>(icb, ocb, block_r);
    break;
  default:
    break;
  }
}
} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_UNTILIZE_LLKS_H

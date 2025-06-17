// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_UNTILIZE_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_UNTILIZE_LLKS_H

namespace experimental {
using std::uint32_t;

#ifdef TRISC_UNPACK
template <bool first_pass = true>
ALWI void llk_unpack_untilize_pass(uint32_t operand, uint32_t block_tile_cols,
                                   uint32_t start_tile_index = 0) {
  const uint32_t operand_id = get_operand_id(operand);
  const uint32_t base_address =
      get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
  const uint32_t page_bytes = get_local_cb_interface(operand_id).fifo_page_size;

  _llk_unpack_untilize_pass_<first_pass>(
      base_address + (start_tile_index * page_bytes), block_tile_cols);
}

ALWI void llk_unpack_untilize(uint32_t operand, uint32_t block_c_tiles,
                              uint32_t start_tile_index = 0) {
  WAYPOINT("UPUW");
  llk_unpack_untilize_pass<true>(operand, block_c_tiles, start_tile_index);
  llk_unpack_untilize_pass<false>(operand, block_c_tiles, start_tile_index);
  WAYPOINT("UPUD");
}
#endif // TRISC_UNPACK

ALWI void untilize_block(uint32_t icb, uint32_t ocb, uint32_t block_r,
                         uint32_t block_c) {
  uint32_t start_tile_idx = 0;
  for (uint32_t i = 0; i < block_r; i++) {
    UNPACK((llk_unpack_untilize(icb, block_c, start_tile_idx)));

    for (uint32_t t = 0; t < block_c; t++) {
      MATH((llk_math_wait_for_dest_available()));

      // Datacopy
      MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE,
                                            BroadcastType::NONE>(0)));

      MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));

      PACK((llk_packer_wait_for_math_done()));

      // Datacopy
      PACK((llk_pack<DST_ACCUM_MODE, false, false>(0, ocb)));

      // Release dest
      PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
    start_tile_idx += block_c;
  }
}
} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_UNTILIZE_LLKS_H

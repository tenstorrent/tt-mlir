// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PACK_UNTILIZE_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PACK_UNTILIZE_LLKS_H

namespace experimental {
using std::uint32_t;

#ifdef TRISC_PACK
template <uint32_t cols_per_dst_pass, uint32_t total_col_tiles>
ALWI void llk_pack_untilize(uint32_t output, uint32_t num_col_tiles_in_pass,
                            uint32_t start_tile_row = 0,
                            uint32_t block_c_index = 0) {
  const uint32_t output_id = get_output_id(output);
  const uint32_t page_size = get_local_cb_interface(output_id).fifo_page_size;
  const uint32_t offset = start_tile_row * total_col_tiles * page_size;

  get_local_cb_interface(output_id).fifo_wr_ptr += offset;
  ::llk_pack_untilize<cols_per_dst_pass, total_col_tiles>(1, output, FACE_R_DIM,
                                                          4, block_c_index, 0);
  get_local_cb_interface(output_id).fifo_wr_ptr -= offset;
}
#endif // TRISC_PACK

template <uint32_t cols_per_dst_pass, uint32_t total_col_tiles>
ALWI void pack_untilize_block(uint32_t icb, uint32_t ocb,
                              uint32_t block_row_tiles,
                              uint32_t block_col_tiles) {
  const uint32_t num_col_blocks = block_col_tiles / cols_per_dst_pass;
  for (uint32_t r = 0; r < block_row_tiles; ++r) {
    for (uint32_t b = 0; b < num_col_blocks; ++b) {
      MATH((llk_math_wait_for_dest_available()));

      for (uint32_t c = 0; c < cols_per_dst_pass; ++c) {
        UNPACK((llk_unpack_A<BroadcastType::NONE, false,
                             EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            icb, r * block_col_tiles + b * cols_per_dst_pass + c)));
        MATH((llk_math_eltwise_unary_datacopy<
              A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(c)));
      }

      MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));

      PACK((llk_packer_wait_for_math_done()));
      PACK((llk_pack_untilize<cols_per_dst_pass, total_col_tiles>(
          ocb, cols_per_dst_pass, r, b)));
      PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
  }
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PACK_UNTILIZE_LLKS_H

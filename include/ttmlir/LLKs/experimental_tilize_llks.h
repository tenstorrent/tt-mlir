// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

STRINGIFY(

namespace experimental {
ALWI void llk_unpack_tilize_block(std::uint32_t operand,
                                  std::uint32_t block_c_tiles,
                                  std::uint32_t start_tile_idx) {
  for (std::uint32_t tile_index = start_tile_idx;
       tile_index < start_tile_idx + block_c_tiles; tile_index++) {
    UNPACK(llk_unpack_tilize(operand, tile_index, block_c_tiles));
}
}

ALWI void tilize_block(uint32_t icb, uint32_t ocb, uint32_t block_r,
                       uint32_t block_c) {
  uint32_t start_tile_idx = 0;
  for (uint32_t i = 0; i < block_r; i++) {
    UNPACK((llk_unpack_tilize_block(icb, block_c, start_tile_idx)));

    for (uint32_t t = 0; t < block_c; t++) {
      // Acquire dst
      MATH((llk_math_wait_for_dest_available()));
      PACK((llk_packer_wait_for_math_done()));

      // Datacopy
      MATH((llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE,
                                            DST_ACCUM_MODE, UnpackToDestEn>(
          0 /*dst index*/)));
      PACK((llk_pack<false, false>(0 /*tile index*/, ocb)));

      // Release dest
      MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
      PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
    start_tile_idx += block_c;
  }
}
}

)

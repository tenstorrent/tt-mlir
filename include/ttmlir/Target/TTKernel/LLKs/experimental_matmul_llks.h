// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MATMUL_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MATMUL_LLKS_H

namespace experimental {

ALWI void matmul_block(uint32_t in0_cb_id, uint32_t in1_cb_id,
                       uint32_t in0_tile_index, uint32_t in1_tile_index,
                       uint32_t idst, const uint32_t transpose, uint32_t ct_dim,
                       uint32_t rt_dim, uint32_t kt_dim,
                       uint32_t in1_k_stride) {
  if (transpose) {
    ckernel::matmul_block_init(in0_cb_id, in1_cb_id, transpose, 1, 1, 1);
    for (uint32_t r = 0; r < rt_dim; r++) {
      for (uint32_t c = 0; c < ct_dim; c++) {
        uint32_t out_tile_index = idst + r * ct_dim + c;
        for (uint32_t k = 0; k < kt_dim; k++) {
          uint32_t a_tile_index = in0_tile_index + r * kt_dim + k;
          uint32_t b_tile_index =
              in1_tile_index + c * kt_dim + k * in1_k_stride;
          ckernel::matmul_block(in0_cb_id, in1_cb_id, a_tile_index,
                                b_tile_index, out_tile_index, transpose, 1, 1,
                                1);
        }
      }
    }
    return;
  }

  for (uint32_t i = 0; i < kt_dim; i++) {
    ckernel::matmul_block(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index,
                          idst, transpose, ct_dim, rt_dim, kt_dim);
    in0_tile_index++;
    in1_tile_index += in1_k_stride;
  }
}

} // namespace experimental

#endif

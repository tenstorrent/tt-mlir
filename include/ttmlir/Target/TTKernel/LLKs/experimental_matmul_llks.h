// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MATMUL_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MATMUL_LLKS_H

namespace experimental {

ALWI void matmul_block(uint32_t in0_cb_id, uint32_t in1_cb_id,
                       uint32_t in0_tile_index, uint32_t in1_tile_index,
                       uint32_t idst, const uint32_t transpose, uint32_t ct_dim,
                       uint32_t rt_dim, uint32_t kt_dim, uint32_t nt_dim) {

  for (uint32_t i = 0; i < kt_dim; i++) {
    ckernel::matmul_block(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index,
                          idst, transpose, ct_dim, rt_dim, kt_dim);
    in0_tile_index++;
    in1_tile_index += nt_dim;
  }
}

} // namespace experimental

#endif

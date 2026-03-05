// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_SCATTER_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_SCATTER_LLKS_H

#include "api/compute/cb_api.h"

namespace experimental {
using std::int32_t;
using std::uint32_t;

constexpr uint32_t SCATTER_TILE_DIM = 32;
constexpr uint32_t SCATTER_MAX_COL_TILES = 8;
constexpr uint32_t SCATTER_MAX_ROW_ELEMS =
    SCATTER_TILE_DIM * SCATTER_MAX_COL_TILES;

inline uint32_t scatter_tile_offset(uint32_t row, uint32_t col) {
  uint32_t fi = row >> 4;
  uint32_t fj = col >> 4;
  uint32_t fk = row & 0xF;
  uint32_t fl = col & 0xF;
  return ((fi * 2 + fj) << 8) + (fk << 4) + fl;
}

inline void scatter_read_row(uint32_t cb_base, uint32_t page_size,
                             uint32_t col_tiles, uint32_t row,
                             volatile uint32_t *dst) {
  for (uint32_t ct = 0; ct < col_tiles; ct++) {
    volatile tt_l1_ptr uint32_t *tile =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(cb_base +
                                                        ct * page_size);
    for (uint32_t c = 0; c < SCATTER_TILE_DIM; c++) {
      dst[ct * SCATTER_TILE_DIM + c] = tile[scatter_tile_offset(row, c)];
    }
  }
}

inline void scatter_write_row(volatile uint32_t *src, uint32_t cb_base,
                              uint32_t page_size, uint32_t col_tiles,
                              uint32_t row) {
  for (uint32_t ct = 0; ct < col_tiles; ct++) {
    volatile tt_l1_ptr uint32_t *tile =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(cb_base +
                                                        ct * page_size);
    for (uint32_t c = 0; c < SCATTER_TILE_DIM; c++) {
      tile[scatter_tile_offset(row, c)] = src[ct * SCATTER_TILE_DIM + c];
    }
  }
}

inline float scatter_u32_to_f32(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = bits;
  return conv.f;
}

inline uint32_t scatter_f32_to_u32(float val) {
  union {
    float f;
    uint32_t u;
  } conv;
  conv.f = val;
  return conv.u;
}

ALWI void scatter_row_block(uint32_t in_cb, uint32_t idx_cb, uint32_t src_cb,
                            uint32_t out_cb, uint32_t in_col_tiles,
                            uint32_t src_col_tiles, uint32_t input_logical_cols,
                            uint32_t src_logical_cols) {
#ifdef TRISC_MATH
  uint32_t in_base = (get_local_cb_interface(in_cb).fifo_rd_ptr) << 4;
  uint32_t idx_base = (get_local_cb_interface(idx_cb).fifo_rd_ptr) << 4;
  uint32_t src_base = (get_local_cb_interface(src_cb).fifo_rd_ptr) << 4;
  uint32_t out_base = (get_local_cb_interface(out_cb).fifo_wr_ptr) << 4;

  uint32_t in_page = get_local_cb_interface(in_cb).fifo_page_size;
  uint32_t idx_page = get_local_cb_interface(idx_cb).fifo_page_size;
  uint32_t src_page = get_local_cb_interface(src_cb).fifo_page_size;
  uint32_t out_page = get_local_cb_interface(out_cb).fifo_page_size;

  uint32_t in_row_elems = in_col_tiles * SCATTER_TILE_DIM;
  uint32_t src_row_elems = src_col_tiles * SCATTER_TILE_DIM;

  for (uint32_t rr = 0; rr < SCATTER_TILE_DIM; rr++) {
    uint32_t row_in[SCATTER_MAX_ROW_ELEMS];
    uint32_t row_idx[SCATTER_MAX_ROW_ELEMS];
    uint32_t row_src[SCATTER_MAX_ROW_ELEMS];

    scatter_read_row(in_base, in_page, in_col_tiles, rr, row_in);
    scatter_read_row(idx_base, idx_page, src_col_tiles, rr, row_idx);
    scatter_read_row(src_base, src_page, src_col_tiles, rr, row_src);

    for (uint32_t s = 0; s < src_logical_cols; s++) {
      float jf = scatter_u32_to_f32(row_idx[s]);
      int32_t j = static_cast<int32_t>(jf);
      if (j >= 0 && static_cast<uint32_t>(j) < input_logical_cols) {
        float old_val = scatter_u32_to_f32(row_in[j]);
        float src_val = scatter_u32_to_f32(row_src[s]);
        row_in[j] = scatter_f32_to_u32(old_val + src_val);
      }
    }

    scatter_write_row(row_in, out_base, out_page, in_col_tiles, rr);
  }
#endif
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_SCATTER_LLKS_H

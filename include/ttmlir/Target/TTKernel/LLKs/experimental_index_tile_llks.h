// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_INDEX_TILE_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_INDEX_TILE_LLKS_H

// This header is for dataflow kernels - it provides functions to write
// pre-computed index tile patterns to CBs for use in partial tile masking.

namespace experimental {

// Convert float to bf16 representation (upper 16 bits of float32).
inline uint16_t float_to_bf16(float val) {
  union {
    float f;
    uint32_t u;
  } conv;
  conv.f = val;
  return static_cast<uint16_t>(conv.u >> 16);
}

// Tile layout for bf16:
// 4 faces (16x16 each), stored in order:
//   Face 0 (i=0, j=0): rows 0-15, cols 0-15
//   Face 1 (i=0, j=1): rows 0-15, cols 16-31
//   Face 2 (i=1, j=0): rows 16-31, cols 0-15
//   Face 3 (i=1, j=1): rows 16-31, cols 16-31
// Within each face: row-major order (k=0..15 rows, l=0..15 cols)
// Each 32-bit word contains 2 bf16 values (low bits = first element)

// Write a row index tile to the specified CB.
// Each element[i,j] = i (the row index, 0-31).
// All columns in a row have the same value.
inline void write_row_index_tile(uint32_t cb_id) {
  cb_reserve_back(cb_id, 1);
  uint32_t write_addr = get_write_ptr(cb_id);
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);

  uint32_t count = 0;
  for (uint32_t i = 0; i < 2; ++i) {      // face_row_half (0=top, 1=bottom)
    for (uint32_t j = 0; j < 2; ++j) {    // face_col_half (0=left, 1=right)
      for (uint32_t k = 0; k < 16; ++k) { // row within face
        // Global row index
        uint32_t global_row = k + 16 * i;
        uint16_t row_bf16 = float_to_bf16(static_cast<float>(global_row));
        // Same value for both elements in the 32-bit word (same row)
        uint32_t packed = (static_cast<uint32_t>(row_bf16) << 16) |
                          static_cast<uint32_t>(row_bf16);
        // Write 8 pairs (16 columns)
        for (uint32_t l = 0; l < 8; ++l) {
          ptr[count++] = packed;
        }
      }
    }
  }
  cb_push_back(cb_id, 1);
}

// Write a column index tile to the specified CB.
// Each element[i,j] = j (the column index, 0-31).
// All rows in a column have the same value.
inline void write_col_index_tile(uint32_t cb_id) {
  cb_reserve_back(cb_id, 1);
  uint32_t write_addr = get_write_ptr(cb_id);
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);

  uint32_t count = 0;
  for (uint32_t i = 0; i < 2; ++i) {      // face_row_half
    for (uint32_t j = 0; j < 2; ++j) {    // face_col_half
      for (uint32_t k = 0; k < 16; ++k) { // row within face
        // Column indices for this face
        for (uint32_t l = 0; l < 16; l += 2) {
          uint32_t global_col0 = l + 16 * j;
          uint32_t global_col1 = l + 1 + 16 * j;
          uint16_t col0_bf16 = float_to_bf16(static_cast<float>(global_col0));
          uint16_t col1_bf16 = float_to_bf16(static_cast<float>(global_col1));
          // Pack two adjacent column values
          uint32_t packed = (static_cast<uint32_t>(col1_bf16) << 16) |
                            static_cast<uint32_t>(col0_bf16);
          ptr[count++] = packed;
        }
      }
    }
  }
  cb_push_back(cb_id, 1);
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_INDEX_TILE_LLKS_H

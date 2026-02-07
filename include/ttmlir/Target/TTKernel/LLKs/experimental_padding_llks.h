// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H

// Include CB API for pack-side functions (llk_wait_for_free_tiles,
// llk_push_tiles) and get_local_cb_interface
#include "compute_kernel_api/cb_api.h"

namespace experimental {
using std::uint32_t;

// Tile layout: 4 faces of 16x16, ordered as:
//   Face 0 (i=0,j=0): rows 0-15, cols 0-15
//   Face 1 (i=0,j=1): rows 0-15, cols 16-31
//   Face 2 (i=1,j=0): rows 16-31, cols 0-15
//   Face 3 (i=1,j=1): rows 16-31, cols 16-31
//
// Memory layout within tile (for bf16):
// ptr[count] iterates: face_row_half (i), face_col_half (j), row_in_face (k),
// col_in_face (l)

// Write a mask tile to L1 memory at the given address
// This runs on the scalar processor, writing bf16 values directly to memory
inline void _write_mask_tile_to_l1_(volatile uint32_t *ptr, uint32_t validRows,
                                    uint32_t validCols) {
  uint32_t count = 0;
  // F32 format: each element is 4 bytes
  constexpr uint32_t ONE_F32 = 0x3F800000;  // 1.0f in F32
  constexpr uint32_t ZERO_F32 = 0x00000000; // 0.0f in F32

  for (uint32_t i = 0; i < 2; ++i) {      // Face row (0-1)
    for (uint32_t j = 0; j < 2; ++j) {    // Face col (0-1)
      for (uint32_t k = 0; k < 16; ++k) { // Row within face
        uint32_t global_row = k + 16 * i;
        bool row_valid = global_row < validRows;

        for (uint32_t l = 0; l < 16; ++l) { // Col within face
          uint32_t global_col = l + 16 * j;
          bool col_valid = global_col < validCols;
          uint32_t val = (row_valid && col_valid) ? ONE_F32 : ZERO_F32;
          ptr[count++] = val;
        }
      }
    }
  }
}

// Write a row mask tile to L1 in F32 format - all columns same value based on
// row validity. F32 tile: 32x32 elements, 4 faces of 16x16, each element is 4
// bytes (1 uint32_t). Total: 1024 uint32_t words.
inline void _write_row_mask_to_l1_(volatile uint32_t *ptr, uint32_t validRows) {
  uint32_t count = 0;
  constexpr uint32_t ONE_F32 = 0x3F800000;  // 1.0f in F32
  constexpr uint32_t ZERO_F32 = 0x00000000; // 0.0f in F32

  for (uint32_t i = 0; i < 2; ++i) {      // Face row (0-1)
    for (uint32_t j = 0; j < 2; ++j) {    // Face col (0-1)
      for (uint32_t k = 0; k < 16; ++k) { // Row within face
        uint32_t global_row = k + 16 * i;
        uint32_t val = (global_row < validRows) ? ONE_F32 : ZERO_F32;
        for (uint32_t l = 0; l < 16; ++l) { // 16 cols per face row
          ptr[count++] = val;
        }
      }
    }
  }
}

// Write a col mask tile to L1 in F32 format - all rows same pattern based on
// col validity. F32 tile: 32x32 elements, 4 faces of 16x16, each element is 4
// bytes (1 uint32_t). Total: 1024 uint32_t words.
inline void _write_col_mask_to_l1_(volatile uint32_t *ptr, uint32_t validCols) {
  uint32_t count = 0;
  constexpr uint32_t ONE_F32 = 0x3F800000;  // 1.0f in F32
  constexpr uint32_t ZERO_F32 = 0x00000000; // 0.0f in F32

  for (uint32_t i = 0; i < 2; ++i) {        // Face row (0-1)
    for (uint32_t j = 0; j < 2; ++j) {      // Face col (0-1)
      for (uint32_t k = 0; k < 16; ++k) {   // Row within face
        for (uint32_t l = 0; l < 16; ++l) { // Col within face
          uint32_t global_col = l + 16 * j;
          uint32_t val = (global_col < validCols) ? ONE_F32 : ZERO_F32;
          ptr[count++] = val;
        }
      }
    }
  }
}

// Include fill.h for ckernel::fill_tile
#include "compute_kernel_api/eltwise_unary/fill.h"

// Fill a tile in DST with a constant scalar value.
// Uses the standard fill_tile API from compute_kernel_api.
ALWI void tile_fill(uint32_t dst_index, float value) {
  MATH((ckernel::fill_tile(dst_index, value)));
}

// CB-based mask write functions - write mask patterns directly to CB memory
// Compiled for TRISC_UNPACK so UNPACK thread blocks until mask is written.
// NO cb_reserve_back/push_back - DM kernel handles CB allocation.
// Uses (fifo_rd_ptr - 1) << 4 to match LLK address calculation convention.

ALWI void write_row_mask_tile(uint32_t validRows, uint32_t cb_id) {
#ifdef TRISC_UNPACK
  // Write directly to L1 at CB read pointer location
  // Use (fifo_rd_ptr - 1) to match LLK address calculation convention
  uint32_t write_addr = (get_local_cb_interface(cb_id).fifo_rd_ptr) << 4;
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
  _write_row_mask_to_l1_(ptr, validRows);
#endif
}

ALWI void write_col_mask_tile(uint32_t validCols, uint32_t cb_id) {
#ifdef TRISC_UNPACK
  // Write directly to L1 at CB read pointer location
  uint32_t write_addr = (get_local_cb_interface(cb_id).fifo_rd_ptr) << 4;
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
  _write_col_mask_to_l1_(ptr, validCols);
#endif
}

// =============================================================================
// Index tile functions for arange operations
// =============================================================================

// Helper to convert uint32_t to F32 bit pattern
inline uint32_t _uint_to_f32_bits_(uint32_t val) {
  float f = static_cast<float>(val);
  uint32_t result;
  __builtin_memcpy(&result, &f, sizeof(result));
  return result;
}

// Write a FULL INDEX tile to L1 in F32 format.
// Each element gets its linear index: element[i,j] = i * 32 + j (0-1023)
// This is used for arange operations where we need the global element index
// within a tile.
inline void _write_full_index_to_l1_(volatile uint32_t *ptr) {
  uint32_t count = 0;

  for (uint32_t i = 0; i < 2; ++i) {      // Face row (0-1)
    for (uint32_t j = 0; j < 2; ++j) {    // Face col (0-1)
      for (uint32_t k = 0; k < 16; ++k) { // Row within face
        uint32_t global_row = k + 16 * i;
        for (uint32_t l = 0; l < 16; ++l) { // Col within face
          uint32_t global_col = l + 16 * j;
          uint32_t linear_idx = global_row * 32 + global_col;
          uint32_t val = _uint_to_f32_bits_(linear_idx);
          ptr[count++] = val;
        }
      }
    }
  }
}

ALWI void write_full_index_tile(uint32_t cb_id) {
#ifdef TRISC_UNPACK
  uint32_t write_addr = (get_local_cb_interface(cb_id).fifo_rd_ptr) << 4;
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
  _write_full_index_to_l1_(ptr);
#endif
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H

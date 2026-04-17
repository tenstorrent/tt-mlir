// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H

// Include CB API for pack-side functions (llk_wait_for_free_tiles,
// llk_push_tiles) and get_local_cb_interface
#include "api/compute/cb_api.h"

namespace experimental {
using std::uint16_t;
using std::uint32_t;

constexpr uint32_t ONE_F32 = 0x3F800000;  // 1.0f in IEEE 754
constexpr uint32_t ZERO_F32 = 0x00000000; // 0.0f in IEEE 754

// Tile layout: 4 faces of 16x16, ordered as:
//   Face 0 (i=0,j=0): rows 0-15, cols 0-15
//   Face 1 (i=0,j=1): rows 0-15, cols 16-31
//   Face 2 (i=1,j=0): rows 16-31, cols 0-15
//   Face 3 (i=1,j=1): rows 16-31, cols 16-31
//
// Memory layout within tile:
// ptr[count] iterates: face_row_half (i), face_col_half (j), row_in_face (k),
// col_in_face (l)

// Write a mask tile to L1 memory at the given address in F32 format.
inline void _write_mask_tile_to_l1_(volatile uint32_t *ptr, uint32_t validRows,
                                    uint32_t validCols) {
  uint32_t count = 0;
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

// Write a row mask tile to L1 - all columns same value based on row validity.
// Works for F32 (T=uint32_t, 4096 B/tile) and BF16 (T=uint16_t, 2048 B/tile).
// Total: 1024 elements of type T.
template <typename T>
inline void _write_row_mask_to_l1_(volatile T *ptr, uint32_t validRows,
                                   T oneVal) {
  uint32_t count = 0;
  for (uint32_t i = 0; i < 2; ++i) {      // Face row (0-1)
    for (uint32_t j = 0; j < 2; ++j) {    // Face col (0-1)
      for (uint32_t k = 0; k < 16; ++k) { // Row within face
        uint32_t global_row = k + 16 * i;
        T val = (global_row < validRows) ? oneVal : T{0};
        for (uint32_t l = 0; l < 16; ++l) { // 16 cols per face row
          ptr[count++] = val;
        }
      }
    }
  }
}

// Write a col mask tile to L1 - all rows same pattern based on col validity.
// Works for F32 (T=uint32_t, 4096 B/tile) and BF16 (T=uint16_t, 2048 B/tile).
// Total: 1024 elements of type T.
template <typename T>
inline void _write_col_mask_to_l1_(volatile T *ptr, uint32_t validCols,
                                   T oneVal) {
  uint32_t count = 0;
  for (uint32_t i = 0; i < 2; ++i) {        // Face row (0-1)
    for (uint32_t j = 0; j < 2; ++j) {      // Face col (0-1)
      for (uint32_t k = 0; k < 16; ++k) {   // Row within face
        for (uint32_t l = 0; l < 16; ++l) { // Col within face
          uint32_t global_col = l + 16 * j;
          T val = (global_col < validCols) ? oneVal : T{0};
          ptr[count++] = val;
        }
      }
    }
  }
}

// CB-based mask write functions - write mask patterns directly to CB memory.
// Compiled for TRISC_UNPACK so UNPACK thread blocks until mask is written.
// NO cb_reserve_back/push_back - DM kernel handles CB allocation.
// Uses fifo_rd_ptr << 4 to match LLK address calculation convention.
//
// Templated on DataFormat so the mask is written in the same format as the CB's
// tile data. Float32 writes 4-byte elements (4096 B/tile); Float16_b writes
// 2-byte elements (2048 B/tile).

constexpr uint16_t ONE_BF16 = 0x3F80; // 1.0 in bfloat16

template <DataFormat df = DataFormat::Float32>
ALWI void write_row_mask_tile(uint32_t validRows, uint32_t cb_id) {
  static_assert(df == DataFormat::Float32 || df == DataFormat::Float16_b,
                "write_row_mask_tile: unsupported DataFormat");
#ifdef TRISC_UNPACK
  uint32_t write_addr = (get_local_cb_interface(cb_id).fifo_rd_ptr) << 4;
  if constexpr (df == DataFormat::Float32) {
    volatile tt_l1_ptr uint32_t *ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
    _write_row_mask_to_l1_(ptr, validRows, ONE_F32);
  } else if constexpr (df == DataFormat::Float16_b) {
    volatile tt_l1_ptr uint16_t *ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t *>(write_addr);
    _write_row_mask_to_l1_(ptr, validRows, ONE_BF16);
  }
#endif
}

template <DataFormat df = DataFormat::Float32>
ALWI void write_col_mask_tile(uint32_t validCols, uint32_t cb_id) {
  static_assert(df == DataFormat::Float32 || df == DataFormat::Float16_b,
                "write_col_mask_tile: unsupported DataFormat");
#ifdef TRISC_UNPACK
  uint32_t write_addr = (get_local_cb_interface(cb_id).fifo_rd_ptr) << 4;
  if constexpr (df == DataFormat::Float32) {
    volatile tt_l1_ptr uint32_t *ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
    _write_col_mask_to_l1_(ptr, validCols, ONE_F32);
  } else if constexpr (df == DataFormat::Float16_b) {
    volatile tt_l1_ptr uint16_t *ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t *>(write_addr);
    _write_col_mask_to_l1_(ptr, validCols, ONE_BF16);
  }
#endif
}

// =============================================================================
// Index tile functions for arange operations
// =============================================================================

// Write a FULL INDEX tile to L1: element[i,j] = i * 32 + j (0-1023).
// Templated on DataFormat so the index is encoded in the CB's native format.
// Float32 writes 4-byte IEEE 754 elements; Float16_b writes 2-byte bfloat16
// elements; Int32 writes 4-byte raw integer elements.
template <DataFormat df, typename T>
inline void _fill_arange_tile_to_l1_(volatile T *ptr) {
  uint32_t count = 0;

  for (uint32_t i = 0; i < 2; ++i) {      // Face row (0-1)
    for (uint32_t j = 0; j < 2; ++j) {    // Face col (0-1)
      for (uint32_t k = 0; k < 16; ++k) { // Row within face
        uint32_t global_row = k + 16 * i;
        for (uint32_t l = 0; l < 16; ++l) { // Col within face
          uint32_t linear_idx = global_row * 32 + (l + 16 * j);
          if constexpr (df == DataFormat::Float32) {
            ptr[count++] = float_to_bits(static_cast<float>(linear_idx));
          } else if constexpr (df == DataFormat::Float16_b) {
            ptr[count++] = static_cast<uint16_t>(
                float_to_bits(static_cast<float>(linear_idx)) >> 16);
          } else if constexpr (df == DataFormat::Int32) {
            ptr[count++] = linear_idx;
          }
        }
      }
    }
  }
}

template <DataFormat df = DataFormat::Float32>
ALWI void fill_arange_tile(uint32_t cb_id) {
  static_assert(df == DataFormat::Float32 || df == DataFormat::Float16_b ||
                    df == DataFormat::Int32,
                "fill_arange_tile: unsupported DataFormat");
#ifdef TRISC_UNPACK
  uint32_t write_addr = (get_local_cb_interface(cb_id).fifo_rd_ptr) << 4;
  if constexpr (df == DataFormat::Float32 || df == DataFormat::Int32) {
    auto *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
    _fill_arange_tile_to_l1_<df>(ptr);
  } else if constexpr (df == DataFormat::Float16_b) {
    auto *ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(write_addr);
    _fill_arange_tile_to_l1_<df>(ptr);
  }
#endif
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H

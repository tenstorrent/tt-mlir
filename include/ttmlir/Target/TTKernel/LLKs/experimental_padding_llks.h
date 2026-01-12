// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H

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
  // Pack two bf16 values (1.0 = 0x3F80, 0.0 = 0x0000) into each 32-bit word
  constexpr uint32_t ONE_ONE = 0x3F803F80;   // Two bf16 1.0s
  constexpr uint32_t ZERO_ZERO = 0x00000000; // Two bf16 0.0s
  constexpr uint32_t ONE_ZERO = 0x00003F80;  // (1.0, 0.0) - first is low bits
  constexpr uint32_t ZERO_ONE = 0x3F800000;  // (0.0, 1.0) - second is high bits

  for (uint32_t i = 0; i < 2; ++i) {      // face_row_half
    for (uint32_t j = 0; j < 2; ++j) {    // face_col_half
      for (uint32_t k = 0; k < 16; ++k) { // row_in_face
        uint32_t global_row = k + 16 * i;
        bool row_valid = global_row < validRows;

        for (uint32_t l = 0; l < 16; l += 2) { // col_in_face, 2 at a time
          uint32_t global_col0 = l + 16 * j;
          uint32_t global_col1 = l + 1 + 16 * j;
          bool col0_valid = global_col0 < validCols;
          bool col1_valid = global_col1 < validCols;

          uint32_t packed;
          if (!row_valid) {
            packed = ZERO_ZERO;
          } else if (col0_valid && col1_valid) {
            packed = ONE_ONE;
          } else if (col0_valid && !col1_valid) {
            packed = ONE_ZERO;
          } else if (!col0_valid && col1_valid) {
            packed = ZERO_ONE;
          } else {
            packed = ZERO_ZERO;
          }
          ptr[count++] = packed;
        }
      }
    }
  }
}

// Write a row mask tile to L1 - all columns same value based on row validity
inline void _write_row_mask_to_l1_(volatile uint32_t *ptr, uint32_t validRows) {
  uint32_t count = 0;
  constexpr uint32_t ONE_ONE = 0x3F803F80;
  constexpr uint32_t ZERO_ZERO = 0x00000000;

  for (uint32_t i = 0; i < 2; ++i) {
    for (uint32_t j = 0; j < 2; ++j) {
      for (uint32_t k = 0; k < 16; ++k) {
        uint32_t global_row = k + 16 * i;
        uint32_t packed = (global_row < validRows) ? ONE_ONE : ZERO_ZERO;
        for (uint32_t l = 0; l < 8; ++l) { // 8 pairs of bf16 = 16 cols
          ptr[count++] = packed;
        }
      }
    }
  }
}

// Write a col mask tile to L1 - all rows same pattern based on col validity
inline void _write_col_mask_to_l1_(volatile uint32_t *ptr, uint32_t validCols) {
  uint32_t count = 0;
  constexpr uint32_t ONE_ONE = 0x3F803F80;
  constexpr uint32_t ZERO_ZERO = 0x00000000;
  constexpr uint32_t ONE_ZERO = 0x00003F80;
  constexpr uint32_t ZERO_ONE = 0x3F800000;

  for (uint32_t i = 0; i < 2; ++i) {
    for (uint32_t j = 0; j < 2; ++j) {
      for (uint32_t k = 0; k < 16; ++k) {
        for (uint32_t l = 0; l < 16; l += 2) {
          uint32_t global_col0 = l + 16 * j;
          uint32_t global_col1 = l + 1 + 16 * j;
          bool col0_valid = global_col0 < validCols;
          bool col1_valid = global_col1 < validCols;

          uint32_t packed;
          if (col0_valid && col1_valid) {
            packed = ONE_ONE;
          } else if (col0_valid && !col1_valid) {
            packed = ONE_ZERO;
          } else if (!col0_valid && col1_valid) {
            packed = ZERO_ONE;
          } else {
            packed = ZERO_ZERO;
          }
          ptr[count++] = packed;
        }
      }
    }
  }
}

// Include fill.h for ckernel::fill_tile (has its own MATH() guards)
#include "compute_kernel_api/eltwise_unary/fill.h"

// Internal SFPU implementations - only compiled for math kernel
#ifdef TRISC_MATH
#include "cmath_common.h"
#include "llk_math_common_api.h"

namespace internal {
using namespace sfpi;

// Generate a row mask tile - linear iteration through 32 rows
// Each row: write same value to all 32 columns
ALWI void _row_mask_sfpu(uint32_t dst_index, uint32_t validRows) {
  math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(
      dst_index);
  math::set_addr_mod_base();
  TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

  // 32 rows, each row has 4 groups of 8 columns
#pragma GCC unroll 0
  for (int row = 0; row < 32; row++) {
    vFloat val = (row < (int)validRows) ? 1.0f : 0.0f;
#pragma GCC unroll 0
    for (int col_group = 0; col_group < 4; col_group++) {
      dst_reg[0] = val;
      dst_reg++;
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 4, 0, 0, p_setrwc::SET_D);
  }

  math::clear_dst_reg_addr();
  TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
  math::clear_addr_mod_base();
}

// Generate a col mask tile - SFPU writes same value to all 8 lanes
// So col granularity is limited to groups of 8
ALWI void _col_mask_sfpu(uint32_t dst_index, uint32_t validCols) {
  math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(
      dst_index);
  math::set_addr_mod_base();
  TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

  // 32 rows, each row has 4 groups of 8 columns
#pragma GCC unroll 0
  for (int row = 0; row < 32; row++) {
#pragma GCC unroll 0
    for (int col_group = 0; col_group < 4; col_group++) {
      int col_start = col_group * 8;
      // All 8 lanes get same value - valid if col_start < validCols
      vFloat val = (col_start < (int)validCols) ? 1.0f : 0.0f;
      dst_reg[0] = val;
      dst_reg++;
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 4, 0, 0, p_setrwc::SET_D);
  }

  math::clear_dst_reg_addr();
  TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
  math::clear_addr_mod_base();
}

// Generate a rectangular padding mask - row AND col constraints
ALWI void _padding_mask_sfpu(uint32_t dst_index, uint32_t validRows,
                             uint32_t validCols) {
  math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(
      dst_index);
  math::set_addr_mod_base();
  TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

  // 32 rows, each row has 4 groups of 8 columns
#pragma GCC unroll 0
  for (int row = 0; row < 32; row++) {
    bool row_valid = (row < (int)validRows);
#pragma GCC unroll 0
    for (int col_group = 0; col_group < 4; col_group++) {
      int col_start = col_group * 8;
      bool col_valid = (col_start < (int)validCols);
      vFloat val = (row_valid && col_valid) ? 1.0f : 0.0f;
      dst_reg[0] = val;
      dst_reg++;
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 4, 0, 0, p_setrwc::SET_D);
  }

  math::clear_dst_reg_addr();
  TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
  math::clear_addr_mod_base();
}

} // namespace internal
#endif // TRISC_MATH

// Public API - wrapped in MATH() macro like standard compute_kernel_api
// These expand to the implementation on math core, no-op on pack/unpack
ALWI void row_mask(uint32_t dst_index, uint32_t validRows) {
  MATH((internal::_row_mask_sfpu(dst_index, validRows)));
}

ALWI void col_mask(uint32_t dst_index, uint32_t validCols) {
  MATH((internal::_col_mask_sfpu(dst_index, validCols)));
}

ALWI void padding_mask(uint32_t dst_index, uint32_t validRows,
                       uint32_t validCols) {
  MATH((internal::_padding_mask_sfpu(dst_index, validRows, validCols)));
}

// Fill a tile in DST with a constant scalar value.
// Uses the standard fill_tile API from compute_kernel_api.
ALWI void tile_fill(uint32_t dst_index, float value) {
  MATH((ckernel::fill_tile(dst_index, value)));
}

} // namespace experimental

// CB-based mask write functions for datamovement kernels
// These reserve CB space, write the mask pattern, and push to CB
// Only available in datamovement context (not compute kernels)
#if !defined(UCK_CHLKC_MATH) && !defined(UCK_CHLKC_PACK) &&                    \
    !defined(UCK_CHLKC_UNPACK)

inline void write_row_mask_tile(uint32_t validRows, uint32_t cb_id) {
  cb_reserve_back(cb_id, 1);
  uint32_t write_addr = get_write_ptr(cb_id);
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
  experimental::_write_row_mask_to_l1_(ptr, validRows);
  cb_push_back(cb_id, 1);
}

inline void write_col_mask_tile(uint32_t validCols, uint32_t cb_id) {
  cb_reserve_back(cb_id, 1);
  uint32_t write_addr = get_write_ptr(cb_id);
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
  experimental::_write_col_mask_to_l1_(ptr, validCols);
  cb_push_back(cb_id, 1);
}

inline void write_combined_mask_tile(uint32_t validRows, uint32_t validCols,
                                     uint32_t cb_id) {
  cb_reserve_back(cb_id, 1);
  uint32_t write_addr = get_write_ptr(cb_id);
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
  experimental::_write_mask_tile_to_l1_(ptr, validRows, validCols);
  cb_push_back(cb_id, 1);
}

#endif // !UCK_CHLKC_MATH && !UCK_CHLKC_PACK && !UCK_CHLKC_UNPACK

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H

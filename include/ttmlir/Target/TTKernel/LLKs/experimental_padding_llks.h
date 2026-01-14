// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H

// Include CB API for pack-side functions (llk_wait_for_free_tiles, llk_push_tiles)
// and get_local_cb_interface
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

  for (uint32_t i = 0; i < 2; ++i) {       // Face row (0-1)
    for (uint32_t j = 0; j < 2; ++j) {     // Face col (0-1)
      for (uint32_t k = 0; k < 16; ++k) {  // Row within face
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

  for (uint32_t i = 0; i < 2; ++i) {       // Face row (0-1)
    for (uint32_t j = 0; j < 2; ++j) {     // Face col (0-1)
      for (uint32_t k = 0; k < 16; ++k) {  // Row within face
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

  for (uint32_t i = 0; i < 2; ++i) {       // Face row (0-1)
    for (uint32_t j = 0; j < 2; ++j) {     // Face col (0-1)
      for (uint32_t k = 0; k < 16; ++k) {  // Row within face
        for (uint32_t l = 0; l < 16; ++l) { // Col within face
          uint32_t global_col = l + 16 * j;
          uint32_t val = (global_col < validCols) ? ONE_F32 : ZERO_F32;
          ptr[count++] = val;
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
#include "llk_sfpu_types.h"

namespace internal {
using namespace sfpi;

// DST tile layout: 4 faces of 16x16 each, processed in order 0,1,2,3
// The standard framework calls sfpu_func 4 times (once per face)
// Each call does ITERATIONS=8 writes, with SETRWC(8)+SETRWC(8) between calls
//
// Total: 4 faces × 8 writes = 32 writes × 8 lanes = 256 elements... 
// But a 32x32 tile has 1024 elements!
//
// Looking at VectorMode::RC more carefully:
// - It calls sfpu_func 4 times (not "per face" but "4 blocks of 4 rows each")
// - SETRWC(8,8) after each = advance by 16
// - So sfpu_func writes 8 elements, advance 16, repeat 4 times
// - 8×4 = 32 writes, + 4×16 = 64 advance = 96 total positions covered?
//
// This doesn't make sense for a full tile. Let me look at what's ACTUALLY happening:
// The key insight: SFPU can only fill 1/4 of the tile per "compute phase"
// The framework is designed for operations that are applied per-element
// For fill_tile, it fills 256 elements (1/4 tile), then needs another call
//
// But looking at our usage: we call row_mask(dst_index, validRows) once
// and expect the whole DST[dst_index] tile to be filled.
//
// The standard fill_tile works because it's called from tile_regs_acquire context
// and the framework handles the full tile iteration internally.
//
// Let me try matching the EXACT pattern of the standard VectorMode::RC:
// 4 iterations, each: 8 writes + SETRWC(8) + SETRWC(8)

// Row mask - SIMPLE approach: treat DST as flat 1024-element buffer
// 32 rows × 32 cols = 1024 elements = 128 writes × 8 lanes
ALWI void _row_mask_sfpu(uint32_t dst_index, uint32_t validRows) {
  DPRINT << ">>> _row_mask_sfpu SIMPLE: dst_index=" << dst_index << " validRows=" << validRows << ENDL();
  
  math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
  math::set_addr_mod_base();
  TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

  // Simple linear iteration: 32 rows, 4 col_groups per row (4 × 8 = 32 cols)
  // Total: 128 writes × 8 lanes = 1024 elements
#pragma GCC unroll 0
  for (int row = 0; row < 32; row++) {
    vFloat val = (row < (int)validRows) ? 1.0f : 0.0f;
#pragma GCC unroll 0
    for (int cg = 0; cg < 4; cg++) {
      dst_reg[0] = val;
      dst_reg++;
    }
  }
  
  DPRINT << ">>> _row_mask_sfpu SIMPLE done" << ENDL();

  math::clear_dst_reg_addr();
  TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
  math::clear_addr_mod_base();
}

// Col mask - SIMPLE approach: treat DST as flat buffer
// Each SFPU write covers 8 columns (8 lanes), all get same value
// Col granularity is 8 columns (we can't mask individual cols within a group)
ALWI void _col_mask_sfpu(uint32_t dst_index, uint32_t validCols) {
  DPRINT << ">>> _col_mask_sfpu SIMPLE: dst_index=" << dst_index << " validCols=" << validCols << ENDL();
  
  math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
  math::set_addr_mod_base();
  TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

  // 32 rows, 4 col_groups per row
  // col_group 0: cols 0-7, group 1: cols 8-15, group 2: cols 16-23, group 3: cols 24-31
#pragma GCC unroll 0
  for (int row = 0; row < 32; row++) {
#pragma GCC unroll 0
    for (int cg = 0; cg < 4; cg++) {
      int col_start = cg * 8;  // First col of this group
      vFloat val = (col_start < (int)validCols) ? 1.0f : 0.0f;
      dst_reg[0] = val;
      dst_reg++;
    }
  }

  math::clear_dst_reg_addr();
  TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
  math::clear_addr_mod_base();
}

// Combined mask - SIMPLE approach
ALWI void _padding_mask_sfpu(uint32_t dst_index, uint32_t validRows, uint32_t validCols) {
  math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
  math::set_addr_mod_base();
  TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

#pragma GCC unroll 0
  for (int row = 0; row < 32; row++) {
    bool row_valid = (row < (int)validRows);
#pragma GCC unroll 0
    for (int cg = 0; cg < 4; cg++) {
      int col_start = cg * 8;
      bool col_valid = (col_start < (int)validCols);
      vFloat val = (row_valid && col_valid) ? 1.0f : 0.0f;
      dst_reg[0] = val;
      dst_reg++;
    }
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
  DPRINT << ">>> tile_fill called: dst_index=" << dst_index << " value=" << value << ENDL();
  MATH((ckernel::fill_tile(dst_index, value)));
}

// CB-based mask write functions - write mask patterns directly to CB memory
// Only compiled for TRISC_PACK since fifo_wr_ptr is only valid in the packer thread
// These functions: reserve CB space, write the mask pattern, and push to CB

ALWI void write_row_mask_tile(uint32_t validRows, uint32_t cb_id) {
#ifdef TRISC_PACK
  DPRINT << ">>> write_row_mask_tile: validRows=" << validRows << " cb_id=" << cb_id << ENDL();
  ckernel::cb_reserve_back(cb_id, 1);
  uint32_t write_addr = get_local_cb_interface(cb_id).fifo_wr_ptr << 4;
  DPRINT << ">>> write_row_mask_tile: write_addr=0x" << HEX() << write_addr << ENDL();
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
  _write_row_mask_to_l1_(ptr, validRows);
  // Print first few values written
  DPRINT << ">>> row mask first 4 words: " << HEX() << ptr[0] << " " << ptr[1] << " " << ptr[2] << " " << ptr[3] << ENDL();
  ckernel::cb_push_back(cb_id, 1);
  DPRINT << ">>> write_row_mask_tile: done push_back" << ENDL();
#endif
}

ALWI void write_col_mask_tile(uint32_t validCols, uint32_t cb_id) {
#ifdef TRISC_PACK
  DPRINT << ">>> write_col_mask_tile: validCols=" << validCols << " cb_id=" << cb_id << ENDL();
  ckernel::cb_reserve_back(cb_id, 1);
  uint32_t write_addr = get_local_cb_interface(cb_id).fifo_wr_ptr << 4;
  DPRINT << ">>> write_col_mask_tile: write_addr=0x" << HEX() << write_addr << ENDL();
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
  _write_col_mask_to_l1_(ptr, validCols);
  // Print first few values written
  DPRINT << ">>> col mask first 4 words: " << HEX() << ptr[0] << " " << ptr[1] << " " << ptr[2] << " " << ptr[3] << ENDL();
  ckernel::cb_push_back(cb_id, 1);
  DPRINT << ">>> write_col_mask_tile: done push_back" << ENDL();
#endif
}

ALWI void write_combined_mask_tile(uint32_t validRows, uint32_t validCols,
                                   uint32_t cb_id) {
#ifdef TRISC_PACK
  ckernel::cb_reserve_back(cb_id, 1);
  uint32_t write_addr = get_local_cb_interface(cb_id).fifo_wr_ptr << 4;
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(write_addr);
  _write_mask_tile_to_l1_(ptr, validRows, validCols);
  ckernel::cb_push_back(cb_id, 1);
#endif
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H

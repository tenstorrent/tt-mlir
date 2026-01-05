// RUN: ttmlir-opt --ttcore-register-device --d2m-decompose-masking -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test d2m-decompose-masking pass: decompose block_mask into linalg.generic
// with per-tile masking using TileWhereOp for efficient element-wise selection.

// CHECK-LABEL: func.func @decompose_block_mask_zero
// Verify block_mask with <zero> fill is decomposed into linalg.generic
// CHECK-DAG: arith.constant 32 : index
// CHECK-DAG: arith.constant 0.000000e+00 : f32
func.func @decompose_block_mask_zero(%input: memref<2x2x!ttcore.tile<32x32, f32>>,
                                     %output: memref<2x2x!ttcore.tile<32x32, f32>>) {
    %c50 = arith.constant 50 : index
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.block_mask
    // CHECK-DAG: linalg.index 0
    // CHECK-DAG: linalg.index 1
    // CHECK: arith.muli
    // CHECK: arith.select
    // CHECK: d2m.tile_sub
    // CHECK: d2m.tile_ltz
    // CHECK: d2m.tile_mul
    // CHECK: d2m.tile_add
<<<<<<< HEAD
    %result = d2m.block_mask %input, %output, %c50, %c50, <zero>
=======
    // CHECK: d2m.tile_where
    d2m.block_mask %input, %output, %c50, %c50, <zero>
>>>>>>> 831a0753d (draft)
        : (memref<2x2x!ttcore.tile<32x32, f32>>, memref<2x2x!ttcore.tile<32x32, f32>>)
        -> memref<2x2x!ttcore.tile<32x32, f32>>
    return
}

// CHECK-LABEL: func.func @decompose_block_mask_neginf
// Verify block_mask with <neginf> fill for max reductions
// CHECK-DAG: arith.constant 0xFF800000 : f32
func.func @decompose_block_mask_neginf(%input: memref<2x2x!ttcore.tile<32x32, f32>>,
                                       %output: memref<2x2x!ttcore.tile<32x32, f32>>) {
    %c50 = arith.constant 50 : index
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.block_mask
    // CHECK: arith.select
    // CHECK: d2m.tile_mul
    // CHECK: d2m.tile_add
<<<<<<< HEAD
    %result = d2m.block_mask %input, %output, %c50, %c50, <neginf>
=======
    // CHECK: d2m.tile_where
    d2m.block_mask %input, %output, %c50, %c50, <neginf>
>>>>>>> 831a0753d (draft)
        : (memref<2x2x!ttcore.tile<32x32, f32>>, memref<2x2x!ttcore.tile<32x32, f32>>)
        -> memref<2x2x!ttcore.tile<32x32, f32>>
    return
}

// CHECK-LABEL: func.func @decompose_block_mask_complete_tile_oob
// Test with logical shape = [32, 32] aligned to 64x64 - tiles at (0,1), (1,0), (1,1)
// are entirely outside the logical bounds. TileWhereOp handles this naturally
// since the mask will be all zeros for those tiles.
func.func @decompose_block_mask_complete_tile_oob(%input: memref<2x2x!ttcore.tile<32x32, f32>>,
                                                   %output: memref<2x2x!ttcore.tile<32x32, f32>>) {
    %c32 = arith.constant 32 : index
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.block_mask
    // CHECK: arith.select
    // CHECK: d2m.tile_mul
    // CHECK: d2m.tile_add
    // CHECK: d2m.tile_where
    d2m.block_mask %input, %output, %c32, %c32, <zero>
        : (memref<2x2x!ttcore.tile<32x32, f32>>, memref<2x2x!ttcore.tile<32x32, f32>>)
        -> memref<2x2x!ttcore.tile<32x32, f32>>
    return
}

// CHECK-LABEL: func.func @decompose_block_mask_dynamic_bounds
// Test with dynamic logical bounds (Index values from function args)
func.func @decompose_block_mask_dynamic_bounds(%input: memref<2x2x!ttcore.tile<32x32, f32>>,
                                               %output: memref<2x2x!ttcore.tile<32x32, f32>>,
                                               %rows: index, %cols: index) {
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.block_mask
    // CHECK: arith.select
    // CHECK: d2m.tile_mul
    // CHECK: d2m.tile_add
    // CHECK: d2m.tile_where
    d2m.block_mask %input, %output, %rows, %cols, <zero>
        : (memref<2x2x!ttcore.tile<32x32, f32>>, memref<2x2x!ttcore.tile<32x32, f32>>)
        -> memref<2x2x!ttcore.tile<32x32, f32>>
    return
}

// CHECK-LABEL: func.func @decompose_block_mask_partial_tile
// Test partial tile masking: logical shape 50x50 means tile (0,0) has partial data
// This should generate index tile loads and per-element masking operations
func.func @decompose_block_mask_partial_tile(%input: memref<2x2x!ttcore.tile<32x32, f32>>,
                                             %output: memref<2x2x!ttcore.tile<32x32, f32>>) {
    %c50 = arith.constant 50 : index
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.block_mask
    // CHECK-DAG: memref.get_global @__d2m_row_index_tile
    // CHECK-DAG: memref.get_global @__d2m_col_index_tile
    // CHECK-DAG: d2m.tile_tilize_block
    // CHECK-DAG: memref.load
    // CHECK-DAG: arith.subi
    // CHECK-DAG: arith.cmpi
    // CHECK-DAG: arith.select
    // CHECK-DAG: d2m.tile_sub
    // CHECK-DAG: d2m.tile_ltz
    // CHECK-DAG: d2m.tile_mul
    // CHECK-DAG: d2m.tile_add
    // CHECK: d2m.tile_where
    d2m.block_mask %input, %output, %c50, %c50, <zero>
        : (memref<2x2x!ttcore.tile<32x32, f32>>, memref<2x2x!ttcore.tile<32x32, f32>>)
    return
}

// CHECK-LABEL: func.func @decompose_block_mask_partial_row_only
// Test partial row masking: logical shape 50x64 means only row dimension is partial
func.func @decompose_block_mask_partial_row_only(%input: memref<2x2x!ttcore.tile<32x32, f32>>,
                                                  %output: memref<2x2x!ttcore.tile<32x32, f32>>) {
    %c50 = arith.constant 50 : index
    %c64 = arith.constant 64 : index
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.block_mask
    // CHECK-DAG: memref.get_global @__d2m_row_index_tile
    // CHECK-DAG: d2m.tile_ltz
    // CHECK: d2m.tile_where
    d2m.block_mask %input, %output, %c50, %c64, <zero>
        : (memref<2x2x!ttcore.tile<32x32, f32>>, memref<2x2x!ttcore.tile<32x32, f32>>)
    return
}

// CHECK-LABEL: func.func @decompose_block_mask_partial_col_only
// Test partial col masking: logical shape 64x50 means only col dimension is partial
func.func @decompose_block_mask_partial_col_only(%input: memref<2x2x!ttcore.tile<32x32, f32>>,
                                                  %output: memref<2x2x!ttcore.tile<32x32, f32>>) {
    %c64 = arith.constant 64 : index
    %c50 = arith.constant 50 : index
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.block_mask
    // CHECK-DAG: memref.get_global @__d2m_col_index_tile
    // CHECK-DAG: d2m.tile_ltz
    // CHECK: d2m.tile_where
    d2m.block_mask %input, %output, %c64, %c50, <zero>
        : (memref<2x2x!ttcore.tile<32x32, f32>>, memref<2x2x!ttcore.tile<32x32, f32>>)
    return
}

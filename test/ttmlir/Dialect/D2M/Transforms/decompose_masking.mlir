// RUN: ttmlir-opt --ttcore-register-device --d2m-decompose-masking -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test d2m-decompose-masking pass: decompose block_mask into linalg.generic
// with per-tile masking using primitive tile arithmetic operations.

// CHECK-LABEL: func.func @decompose_block_mask_zero
// Verify block_mask with <zero> fill is decomposed into linalg.generic
// Constants may be hoisted outside linalg.generic by canonicalization
// CHECK-DAG: arith.constant 32 : index
// CHECK-DAG: arith.constant 0.000000e+00 : f32
// CHECK-DAG: arith.constant 1.000000e+00 : f32
func.func @decompose_block_mask_zero(%input: memref<2x2x!ttcore.tile<32x32, f32>>,
                                     %output: memref<2x2x!ttcore.tile<32x32, f32>>) {
    %c50 = arith.constant 50 : index
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.block_mask
    // CHECK-DAG: linalg.index 0
    // CHECK-DAG: linalg.index 1
    // CHECK: arith.muli
    // CHECK: arith.cmpi sge
    // CHECK: arith.ori
    // CHECK: arith.select
    // CHECK: d2m.tile_mul
    // CHECK: d2m.tile_add
    %result = d2m.block_mask %input, %output, %c50, %c50, <zero>
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
    %result = d2m.block_mask %input, %output, %c50, %c50, <neginf>
        : (memref<2x2x!ttcore.tile<32x32, f32>>, memref<2x2x!ttcore.tile<32x32, f32>>)
        -> memref<2x2x!ttcore.tile<32x32, f32>>
    return
}

// CHECK-LABEL: func.func @decompose_block_mask_complete_tile_oob
// Test with logical shape = [32, 32] aligned to 64x64 - tiles at (0,1), (1,0), (1,1)
// are entirely outside the logical bounds.
func.func @decompose_block_mask_complete_tile_oob(%input: memref<2x2x!ttcore.tile<32x32, f32>>,
                                                   %output: memref<2x2x!ttcore.tile<32x32, f32>>) {
    %c32 = arith.constant 32 : index
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.block_mask
    // CHECK: arith.select
    // CHECK: d2m.tile_mul
    // CHECK: d2m.tile_add
    %result = d2m.block_mask %input, %output, %c32, %c32, <zero>
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
    // CHECK: arith.cmpi sge
    // CHECK: arith.ori
    // CHECK: arith.select
    // CHECK: d2m.tile_mul
    // CHECK: d2m.tile_add
    %result = d2m.block_mask %input, %output, %rows, %cols, <zero>
        : (memref<2x2x!ttcore.tile<32x32, f32>>, memref<2x2x!ttcore.tile<32x32, f32>>)
        -> memref<2x2x!ttcore.tile<32x32, f32>>
    return
}

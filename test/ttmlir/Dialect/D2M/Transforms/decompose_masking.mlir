// RUN: ttmlir-opt --ttcore-register-device --d2m-decompose-masking -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test d2m-decompose-masking pass: decompose tile_mask_boundary into primitive
// tile arithmetic operations within linalg.generic bodies.

// CHECK-LABEL: func.func @decompose_mask_boundary_zero
// Verify tile_mask_boundary with <zero> fill is decomposed
// Constants may be hoisted outside the generic by canonicalization
// CHECK-DAG: arith.constant 50 : index
// CHECK-DAG: arith.constant 0.000000e+00 : f32
// CHECK-DAG: arith.constant 1.000000e+00 : f32
func.func @decompose_mask_boundary_zero(%input: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
    %empty = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.tile_mask_boundary
    // CHECK-DAG: linalg.index 0
    // CHECK-DAG: linalg.index 1
    // CHECK: arith.muli
    // CHECK: arith.cmpi sge
    // CHECK: arith.ori
    // CHECK: arith.select
    // CHECK: d2m.tile_mul
    // CHECK: d2m.tile_add
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%empty : tensor<2x2x!ttcore.tile<32x32, f32>>) {
    ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %masked = d2m.tile_mask_boundary %in, [50, 50], <zero>
            : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
        linalg.yield %masked : !ttcore.tile<32x32, f32>
    } -> tensor<2x2x!ttcore.tile<32x32, f32>>
    return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: func.func @decompose_mask_boundary_neginf
// Verify tile_mask_boundary with <neginf> fill for max reductions
// CHECK-DAG: arith.constant 0xFF800000 : f32
func.func @decompose_mask_boundary_neginf(%input: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
    %empty = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.tile_mask_boundary
    // CHECK: arith.select
    // CHECK: d2m.tile_mul
    // CHECK: d2m.tile_add
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%empty : tensor<2x2x!ttcore.tile<32x32, f32>>) {
    ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %masked = d2m.tile_mask_boundary %in, [50, 50], <neginf>
            : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
        linalg.yield %masked : !ttcore.tile<32x32, f32>
    } -> tensor<2x2x!ttcore.tile<32x32, f32>>
    return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: func.func @decompose_mask_complete_tile_oob
// Test with logical shape = [32, 32] aligned to 64x64 - tiles at (0,1), (1,0), (1,1)
// are entirely outside the logical bounds.
// CHECK-DAG: arith.constant 32 : index
func.func @decompose_mask_complete_tile_oob(%input: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
    %empty = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
    // CHECK: linalg.generic
    // CHECK-NOT: d2m.tile_mask_boundary
    // CHECK: arith.select
    // CHECK: d2m.tile_mul
    // CHECK: d2m.tile_add
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%empty : tensor<2x2x!ttcore.tile<32x32, f32>>) {
    ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %masked = d2m.tile_mask_boundary %in, [32, 32], <zero>
            : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
        linalg.yield %masked : !ttcore.tile<32x32, f32>
    } -> tensor<2x2x!ttcore.tile<32x32, f32>>
    return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns --d2m-grid-selection -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that ttir.matmul with transpose_b is lowered to d2m correctly:
//  - The B operand's affine indexing map on the inner linalg.generic swaps
//    the K/N iteration dimensions (K becomes the innermost, N becomes outer).
//  - The raw d2m.tile_matmul op does not carry a transpose_b attribute.

// transpose_b = false (baseline): RHS map is (K, N).
// CHECK-DAG: #[[$MAP_LHS:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP_RHS:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP_OUT:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_no_transpose
// CHECK: linalg.generic {indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_OUT]]]
// CHECK: "d2m.tile_matmul"
func.func @matmul_no_transpose(%lhs: tensor<128x96xf32>, %rhs: tensor<96x64xf32>) -> tensor<128x64xf32> {
  %r = "ttir.matmul"(%lhs, %rhs) : (tensor<128x96xf32>, tensor<96x64xf32>) -> tensor<128x64xf32>
  return %r : tensor<128x64xf32>
}

// -----

// transpose_b = true: RHS map is (N, K) instead of (K, N).
// CHECK-DAG: #[[$MAP_LHS_T:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP_RHS_T:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MAP_OUT_T:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_transpose_b
// CHECK: linalg.generic {indexing_maps = [#[[$MAP_LHS_T]], #[[$MAP_RHS_T]], #[[$MAP_OUT_T]]]
// CHECK: "d2m.tile_matmul"
func.func @matmul_transpose_b(%lhs: tensor<128x96xf32>, %rhs: tensor<64x96xf32>) -> tensor<128x64xf32> {
  %r = "ttir.matmul"(%lhs, %rhs) <{transpose_b = true}> : (tensor<128x96xf32>, tensor<64x96xf32>) -> tensor<128x64xf32>
  return %r : tensor<128x64xf32>
}

// RUN: ttmlir-opt --ttir-reenable-dps %s | FileCheck %s

// This simulates a broken DPS pattern where the output buffer %arg1 is unused
// and a new empty tensor is created instead
// CHECK-LABEL: func.func @test_broken_dps_simple
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x128xf32>, %[[ARG1:.*]]: tensor<64x128xf32>)
func.func @test_broken_dps_simple(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> attributes {ttir.return_to_output_mapping = 1 : i32} {
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.abs ins(%arg0 : tensor<64x128xf32>) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>

  // CHECK-NOT: tensor.empty()
  // CHECK: linalg.abs ins(%[[ARG0]] : tensor<64x128xf32>) outs(%[[ARG1]] : tensor<64x128xf32>)
  return %1 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_broken_dps_with_reduce_and_expand
// CHECK-SAME: (%{{.*}}: tensor<64x128xf32>, %[[ARG1:.*]]: tensor<1x64x128xf32>)
func.func @test_broken_dps_with_reduce_and_expand(%arg0: tensor<64x128xf32>, %arg1: tensor<1x64x128xf32>) -> tensor<1x64x128xf32> attributes {ttir.return_to_output_mapping = 1 : i32} {
  // Pattern with reduce followed by expand_shape
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>

  %2 = linalg.reduce ins(%arg0 : tensor<64x128xf32>) outs(%1 : tensor<64x128xf32>) dimensions = []
    (%in: f32, %out: f32) {
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
    }

  %3 = tensor.expand_shape %2 [[0, 1], [2]] output_shape [1, 64, 128] : tensor<64x128xf32> into tensor<1x64x128xf32>

  // CHECK-NOT: tensor.empty()
  // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG1]] {{\[\[}}0, 1], [2]] : tensor<1x64x128xf32> into tensor<64x128xf32>
  // CHECK: linalg.fill ins(%{{.*}} : f32) outs(%[[COLLAPSED]] : tensor<64x128xf32>)
  return %3 : tensor<1x64x128xf32>
}

// CHECK-LABEL: func.func @test_noop_case
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x128xf32>, %[[ARG1:.*]]: tensor<64x128xf32>)
func.func @test_noop_case(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> attributes {ttir.return_to_output_mapping = 1 : i32} {
  // NOOP case where the return value is directly a block argument
  // CHECK: linalg.copy ins(%[[ARG0]] : tensor<64x128xf32>) outs(%[[ARG1]] : tensor<64x128xf32>)
  return %arg0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_already_enabled_dps
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x128xf32>, %[[ARG1:.*]]: tensor<64x128xf32>)
func.func @test_already_enabled_dps(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> attributes {ttir.return_to_output_mapping = 1 : i32} {
  // This function already uses the output buffer, so the pass should skip it
  %0 = linalg.abs ins(%arg0 : tensor<64x128xf32>) outs(%arg1 : tensor<64x128xf32>) -> tensor<64x128xf32>

  // CHECK: linalg.abs ins(%[[ARG0]] : tensor<64x128xf32>) outs(%[[ARG1]] : tensor<64x128xf32>)
  // CHECK-NOT: ttir.return_to_output_mapping
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_no_mapping_attr
// CHECK-SAME: (%{{.*}}: tensor<64x128xf32>)
func.func @test_no_mapping_attr(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // Function without the mapping attribute should be unchanged
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.abs ins(%arg0 : tensor<64x128xf32>) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>

  // CHECK: tensor.empty()
  // CHECK: linalg.abs
  return %1 : tensor<64x128xf32>
}

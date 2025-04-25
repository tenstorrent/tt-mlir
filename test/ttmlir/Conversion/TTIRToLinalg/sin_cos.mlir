// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module attributes {} {
  // Test for sine operation
  func.func @sin_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[SIZE:tensor<64x128xf32>]]
    %0 = ttir.empty() : tensor<64x128xf32>

    // CHECK: [[VAL1:%[0-9]+]] = linalg.generic {{{.*}}} ins(%arg0 : tensor<64x128xf32>) outs(%{{.*}} : tensor<64x128xf32>) {
    // CHECK: ^bb0(%in: f32, %out: f32):
    // CHECK:   [[SIN:%[0-9]+]] = math.sin %in : f32
    // CHECK:   linalg.yield [[SIN]] : f32
    // CHECK: } -> tensor<64x128xf32>

    %1 = "ttir.sin"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    return %1 : tensor<64x128xf32>
  }

  // Test for cosine operation
  func.func @cos_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[SIZE:tensor<64x128xf32>]]
    %0 = ttir.empty() : tensor<64x128xf32>

    // CHECK: [[VAL1:%[0-9]+]] = linalg.generic {{{.*}}} ins(%arg0 : tensor<64x128xf32>) outs(%{{.*}} : tensor<64x128xf32>) {
    // CHECK: ^bb0(%in: f32, %out: f32):
    // CHECK:   [[COS:%[0-9]+]] = math.cos %in : f32
    // CHECK:   linalg.yield [[COS]] : f32
    // CHECK: } -> tensor<64x128xf32>

    %1 = "ttir.cos"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    return %1 : tensor<64x128xf32>
  }
}

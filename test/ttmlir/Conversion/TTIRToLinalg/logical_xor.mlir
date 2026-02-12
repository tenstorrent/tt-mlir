// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @logical_xor_test
func.func @logical_xor_test(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.logical_xor
  // CHECK: tosa.select
  %1 = "ttir.logical_xor"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @logical_xor_3d
func.func @logical_xor_3d(%arg0: tensor<2x64x128xf32>, %arg1: tensor<2x64x128xf32>) -> tensor<2x64x128xf32> {
  // CHECK: tosa.logical_xor
  %1 = "ttir.logical_xor"(%arg0, %arg1) : (tensor<2x64x128xf32>, tensor<2x64x128xf32>) -> tensor<2x64x128xf32>
  return %1 : tensor<2x64x128xf32>
}

// CHECK-LABEL: func.func @logical_xor_small
func.func @logical_xor_small(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
  // CHECK: tosa.logical_xor
  %1 = "ttir.logical_xor"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  return %1 : tensor<1x1xf32>
}

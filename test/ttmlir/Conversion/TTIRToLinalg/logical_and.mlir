// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @logical_and_test
func.func @logical_and_test(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.logical_and
  // CHECK: tosa.cast
  %1 = "ttir.logical_and"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @logical_and_3d
func.func @logical_and_3d(%arg0: tensor<2x64x128xf32>, %arg1: tensor<2x64x128xf32>) -> tensor<2x64x128xf32> {
  // CHECK: tosa.logical_and
  %1 = "ttir.logical_and"(%arg0, %arg1) : (tensor<2x64x128xf32>, tensor<2x64x128xf32>) -> tensor<2x64x128xf32>
  return %1 : tensor<2x64x128xf32>
}

// CHECK-LABEL: func.func @logical_and_small
func.func @logical_and_small(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
  // CHECK: tosa.logical_and
  %1 = "ttir.logical_and"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  return %1 : tensor<1x1xf32>
}

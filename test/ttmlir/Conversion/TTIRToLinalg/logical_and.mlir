// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @logical_and_test(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.logical_and
  // CHECK: tosa.select
  %1 = "ttir.logical_and"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

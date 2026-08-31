// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @round_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: math.roundeven
  %1 = "ttir.round"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

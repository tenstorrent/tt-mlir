// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @log_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: [[VAL1:%[0-9]+]] = tosa.log
  %1 = "ttir.log"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: return [[VAL1]]
  return %1 : tensor<64x128xf32>
}

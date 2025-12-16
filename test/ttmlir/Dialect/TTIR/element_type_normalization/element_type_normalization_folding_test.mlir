// RUN: ttmlir-opt --ttir-element-type-normalization -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @fold_broadcast(%arg0 : tensor<1x256x1xf64>) -> tensor<1x256x1xf64> {
  // CHECK-NOT: "ttir.broadcast"
  // CHECK: -> tensor<1x256x1xf32>
  %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x256x1xf64>) -> tensor<1x256x1xf64>
  return %1 : tensor<1x256x1xf64>
}

// RUN: ttmlir-opt --canonicalize --ttir-decompose-complex-reshape %s | FileCheck %s

// The unsqueeze version of @singleton_transpose_append_trailing_one from decompose_complex_reshape.mlir.
// Separate file because --canonicalize folds away the identity reshape that file's @no_transform_identity needs.
// CHECK-LABEL: @singleton_transpose_from_unsqueeze
// CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%arg0)
// CHECK-SAME: (tensor<128xf32>) -> tensor<1x128xf32>
// CHECK: "ttir.permute"(%[[RESHAPE]])
// CHECK-SAME: permutation = array<i64: 1, 0>
// CHECK-NOT: ttir.unsqueeze
func.func @singleton_transpose_from_unsqueeze(%arg0: tensor<128xf32>) -> tensor<128x1xf32> {
  %0 = "ttir.unsqueeze"(%arg0) <{dim = 1 : si32}> : (tensor<128xf32>) -> tensor<128x1xf32>
  return %0 : tensor<128x1xf32>
}

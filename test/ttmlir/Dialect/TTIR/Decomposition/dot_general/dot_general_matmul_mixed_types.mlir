// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s

// Regression test for failure when the inputs are mixed types
// https://github.com/tenstorrent/tt-mlir/issues/6227

func.func @main(%arg0: tensor<1x16x128x512x16xf32>, %arg1: tensor<1x16x128x1x512xbf16>) -> tensor<1x16x128x1x16xf32> {
  %0 = "ttir.dot_general"(%arg1, %arg0) <{batch_dims_lhs = array<i64: 0, 1, 2>, batch_dims_rhs = array<i64: 0, 1, 2>, contract_dims_lhs = array<i64: 4>, contract_dims_rhs = array<i64: 3>}> : (tensor<1x16x128x1x512xbf16>, tensor<1x16x128x512x16xf32>) -> tensor<1x16x128x1x16xf32>
  return %0 : tensor<1x16x128x1x16xf32>
}

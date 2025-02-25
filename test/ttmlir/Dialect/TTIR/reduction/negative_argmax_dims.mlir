// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for argmax op.

// CHECK: error: 'ttir.argmax' op can only reduce one dimension; number of specified dimensions: 2.
func.func public @argmax_3d(%arg0: tensor<128x28x28xf32>) -> tensor<128x28xi32> {
  %0 = tensor.empty() : tensor<128x28xi32>
  %1 = "ttir.argmax"(%arg0, %0) <{dim_arg = [0: i32, 2 : i32], keep_dim = false}> : (tensor<128x28x28xf32>, tensor<128x28xi32>) -> tensor<128x28xi32>
  return %1 : tensor<128x28xi32>
}

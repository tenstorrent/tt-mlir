// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

module  {
  func.func @test_linear_op_rewrite(%arg0: tensor<2x33x1024xf32>, %arg1: tensor<2x1024x1024xf32>, %arg2: tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>{
    // CHECK-LABEL: func.func @test_linear_op_rewrite
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.add"
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<2x33x1024xf32>, tensor<2x1024x1024xf32>, tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>
    return %result : tensor<2x33x1024xf32>
  }
}

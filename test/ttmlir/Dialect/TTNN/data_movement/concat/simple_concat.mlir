// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<32x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<32x96xf32> {
    // CHECK: = "ttnn.concat"
    %1 = "ttir.concat"(%arg0, %arg1) <{dim = 1 : si32}> : (tensor<32x32xf32>, tensor<32x64xf32>) -> tensor<32x96xf32>
    return %1 : tensor<32x96xf32>
  }
}

func.func @concat_zero_dim_in_collapse_range(%arg0: tensor<1x0x1024xf32>, %arg1: tensor<1x8x1024xf32>) -> tensor<1x8x1024xf32> {
  // CHECK-NOT: "ttnn.concat"
  %1 = "ttir.concat"(%arg0, %arg1) <{dim = 1 : si32}> : (tensor<1x0x1024xf32>, tensor<1x8x1024xf32>) -> tensor<1x8x1024xf32>
  return %1 : tensor<1x8x1024xf32>
}

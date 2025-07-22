// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @bcast_one_dim(%arg0: tensor<2x64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<2x64x128xf32> {
    %0 = ttir.empty() : tensor<2x64x128xf32>
    // CHECK: "ttnn.multiply"
    %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<2x64x128xf32>, tensor<64x128xf32>, tensor<2x64x128xf32>) -> tensor<2x64x128xf32>
    return %1 : tensor<2x64x128xf32>
  }

  func.func @bcast_multi_dim(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<15x1xf32>) -> tensor<17x16x15x14xf32> {
    %0 = ttir.empty() : tensor<17x16x15x14xf32>
    // CHECK: "ttnn.multiply"
    %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<17x16x15x14xf32>, tensor<15x1xf32>, tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32>
    return %1 : tensor<17x16x15x14xf32>
  }

}

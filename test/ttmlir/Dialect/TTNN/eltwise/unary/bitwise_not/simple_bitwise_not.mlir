// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @bitwise_not(%arg0: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_not"(%arg0, %0) : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: "ttnn.bitwise_not"
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: -> tensor<64x128xsi32
    return %1 : tensor<64x128xi32>
  }
}

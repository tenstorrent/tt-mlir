// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck --input-file=%t %s

module attributes {} {
  func.func @test_topk(%arg0: tensor<64x128xbf16>) -> (tensor<64x8xbf16>, tensor<64x8xi16>) {
    // CHECK-LABEL: @test_topk
    // CHECK: %{{.*}}, %{{.*}} = "ttnn.topk"(%arg0)
    // CHECK-SAME: <{dim = -1 : si8, k = 8 : ui32, largest = true, sorted = true}>
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: -> (tensor<64x8xbf16,
    // CHECK-SAME: tensor<64x8xui16,
    %0, %1 = "ttir.topk"(%arg0) <{k = 8 : ui32, dim = -1 : si32, largest = true, sorted = true}> : (tensor<64x128xbf16>) -> (tensor<64x8xbf16>, tensor<64x8xi16>)
    return %0, %1 : tensor<64x8xbf16>, tensor<64x8xi16>
  }
}

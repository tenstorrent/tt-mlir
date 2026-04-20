// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @test_basic_top_k(%input: tensor<2x3x32x128xbf16>) -> (tensor<2x3x32x5xbf16>, tensor<2x3x32x5xui16>) {
    // CHECK-LABEL: func.func @test_basic_top_k
    // CHECK: %{{.*}}, %{{.*}} = "ttnn.topk"(%arg0)
    // CHECK-SAME: <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = false}>
    // CHECK-SAME: tensor<2x3x32x128xbf16,
    // CHECK-SAME: -> (tensor<2x3x32x5xbf16,
    // CHECK-SAME: tensor<2x3x32x5xui16,
    %values, %indices = "ttir.topk"(%input) { k = 5 : i32} : (tensor<2x3x32x128xbf16>) -> (tensor<2x3x32x5xbf16>, tensor<2x3x32x5xui16>)
    return %values, %indices : tensor<2x3x32x5xbf16>, tensor<2x3x32x5xui16>
  }
}

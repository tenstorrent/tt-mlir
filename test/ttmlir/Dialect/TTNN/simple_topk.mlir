// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @test_basic_top_k(%input: tensor<2x3x32x128xf32>) -> (tensor<2x3x32x5xf32>, tensor<2x3x32x5xi32>) {
    // CHECK-LABEL: func.func @test_basic_top_k
    // CHECK: %{{.*}}, %{{.*}} = "ttnn.topk"(%arg0)
    // CHECK-SAME: <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = true}>
    // CHECK-SAME: tensor<2x3x32x128xf32,
    // CHECK-SAME: -> (tensor<2x3x32x5xf32,
    // CHECK-SAME: tensor<2x3x32x5xsi32,
    %values, %indices = "ttir.topk"(%input) { k = 5 : i32} : (tensor<2x3x32x128xf32>) -> (tensor<2x3x32x5xf32>, tensor<2x3x32x5xi32>)
    return %values, %indices : tensor<2x3x32x5xf32>, tensor<2x3x32x5xi32>
  }
}

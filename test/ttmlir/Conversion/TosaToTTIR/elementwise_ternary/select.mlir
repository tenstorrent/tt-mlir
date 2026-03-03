// RUN: ttmlir-opt --convert-tosa-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @test_select(%arg0: tensor<32x128xi1>, %arg1: tensor<32x128xf32>, %arg2: tensor<32x128xf32>) -> tensor<32x128xf32> {
    // CHECK: func.func {{.+}} [[SELECTOR:tensor<[0-9]+x[0-9]+xi1>]]
    %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<32x128xi1>, tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    // CHECK: %[[VAL:[0-9]+]] = "ttir.where"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, %arg{{[0-9]+}}) : ([[SELECTOR:tensor<32x128xi1>]], [[TENSOR_SIZE:tensor<32x128xf32>]], [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    // CHECK: return %[[VAL]] : [[TENSOR_SIZE]]
    return %0 : tensor<32x128xf32>
  }
}

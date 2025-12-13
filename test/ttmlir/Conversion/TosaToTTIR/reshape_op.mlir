// RUN: ttmlir-opt --convert-tosa-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
    // CHECK-LABEL: func.func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
    // CHECK-NEXT: [[SHAPE:%.+]] = "ttir.constant"() <{value = dense<[1, 819]> : tensor<2xi64>}> : () -> tensor<2xi64>
    // CHECK-NEXT: [[RESHAPE:%.+]] = "ttir.reshape"(%arg0) <{shape = [1 : i32, 819 : i32]}> : (tensor<13x21x3xf32>) -> tensor<1x819xf32>
    // CHECK-NEXT: return [[RESHAPE]] : tensor<1x819xf32>
    // CHECK-NEXT: }
    %1 = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.reshape %arg0, %1 : (tensor<13x21x3xf32>, !tosa.shape<2>) -> tensor<1x819xf32>
    return %0 : tensor<1x819xf32>
  }
}

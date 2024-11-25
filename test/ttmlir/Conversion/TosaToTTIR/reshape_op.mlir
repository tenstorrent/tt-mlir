// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_reshape(%arg0: tensor<63xf32>) -> tensor<1x3x3x7xf32> {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 3, 3, 7>} : (tensor<63xf32>) -> tensor<1x3x3x7xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.reshape"[[C:.*]]
    return %0 : tensor<1x3x3x7xf32>
  }
}

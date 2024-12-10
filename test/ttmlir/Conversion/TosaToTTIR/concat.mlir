// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_concat(%arg0: tensor<13x17x1xf32>, %arg1: tensor<13x17x1xf32>, %arg2: tensor<13x17x1xf32>) -> tensor<13x51x1xf32> {
    %0 = tosa.concat %arg0, %arg1, %arg2 {axis = 1 : i32} : (tensor<13x17x1xf32>, tensor<13x17x1xf32>, tensor<13x17x1xf32>) -> tensor<13x51x1xf32>
    // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[TENSOR_SIZE:tensor<13x51x1xf32>]]
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.concat"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, %arg{{[0-9]+}}, [[VAL0]]) <{dim = 1 : si32{{.+}}: (tensor<13x17x1xf32>, tensor<13x17x1xf32>, tensor<13x17x1xf32>, [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    return %0 : tensor<13x51x1xf32>
    // CHECK: return [[VAL1]] : [[TENSOR_SIZE]]
  }
}

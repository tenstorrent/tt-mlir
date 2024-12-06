// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_clamp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = tosa.clamp %arg0 { min_int = 2 : i64, max_int = 3 : i64, min_fp = 2.0 : f32, max_fp = 3.0 : f32 } : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = tensor.empty() : [[TENSOR_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+xf[0-9]+>]]
    // CHECK: %[[VAL:[0-9]+]] = "ttir.clamp"(%arg{{[0-9]+}}, %[[OP_OUT]])
    // CHECK-SAME: max = 3.000000e+00 : f32, min = 2.000000e+00 : f32{{.+}}: ([[TENSOR_SIZE]], [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    return %0 : tensor<13x21x3xf32>
    // CHECK: return %[[VAL]] : [[TENSOR_SIZE]]
  }
}

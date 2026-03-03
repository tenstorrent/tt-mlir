// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_max_and_sum_reduce_window(%arg0: tensor<1x14x14x32xbf16>, %arg1: tensor<1x14x14x32xbf16>) -> (tensor<1x7x7x32xbf16>, tensor<1x7x7x32xbf16>) {
  %init_max = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %init_sum = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  // CHECK: "ttir.max_pool2d"
  // CHECK-SAME: kernel = array<i32: 2, 2>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK: "ttir.avg_pool2d"
  // CHECK-SAME: kernel = array<i32: 2, 2>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK: "ttir.multiply"
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %init_max, %init_sum) ({
    ^bb0(%a0: tensor<bf16>, %a1: tensor<bf16>, %b0: tensor<bf16>, %b1: tensor<bf16>):
      %2 = stablehlo.maximum %a0, %b0 : tensor<bf16>
      %3 = stablehlo.add %a1, %b1 : tensor<bf16>
      "stablehlo.return"(%2, %3) : (tensor<bf16>, tensor<bf16>) -> ()
    })
    { padding = dense<0> : tensor<4x2xi64>,
      window_dimensions = array<i64: 1, 2, 2, 1>,
      window_strides = array<i64: 1, 2, 2, 1>}
    : (tensor<1x14x14x32xbf16>, tensor<1x14x14x32xbf16>, tensor<bf16>, tensor<bf16>) -> (tensor<1x7x7x32xbf16>, tensor<1x7x7x32xbf16>)
  func.return %0#0, %0#1 : tensor<1x7x7x32xbf16>, tensor<1x7x7x32xbf16>
}

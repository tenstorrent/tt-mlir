// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-unroll-static-while %s | FileCheck %s

module @static_while {
  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %0:2 = stablehlo.while(%iterArg = %c0, %iterArg_0 = %arg0) : tensor<i32>, tensor<2x2xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg_0, %iterArg_0 : tensor<2x2xf32>
      %2 = stablehlo.add %iterArg, %c1 : tensor<i32>
      stablehlo.return %2, %1 : tensor<i32>, tensor<2x2xf32>
    }
    return %0#1 : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: stablehlo.while
// CHECK: stablehlo.constant dense<0> : tensor<i32>
// CHECK: stablehlo.add
// CHECK: stablehlo.constant dense<1> : tensor<i32>
// CHECK: stablehlo.add
// CHECK: return

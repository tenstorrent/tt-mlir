// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_compare attributes {} {
  func.func public @test_eq(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  EQ, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: = "ttir.eq"(%arg0, %arg1)
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    return %0 : tensor<13x31xi1>
    // CHECK: return %0 : tensor<13x31xi1>
  }

  func.func public @test_ne(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  NE, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: = "ttir.ne"(%arg0, %arg1)
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    return %0 : tensor<13x31xi1>
    // CHECK: return %0 : tensor<13x31xi1>
  }

  func.func public @test_ge(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  GE, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: = "ttir.ge"(%arg0, %arg1)
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    return %0 : tensor<13x31xi1>
    // CHECK: return %0 : tensor<13x31xi1>
  }

  func.func public @test_gt(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  GT, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: = "ttir.gt"(%arg0, %arg1)
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    return %0 : tensor<13x31xi1>
    // CHECK: return %0 : tensor<13x31xi1>
  }

  func.func public @test_le(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  LE, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: = "ttir.le"(%arg0, %arg1)
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    return %0 : tensor<13x31xi1>
    // CHECK: return %0 : tensor<13x31xi1>
  }

  func.func public @test_lt(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  LT, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: = "ttir.lt"(%arg0, %arg1)
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    return %0 : tensor<13x31xi1>
    // CHECK: return %0 : tensor<13x31xi1>
  }

  // Frontends (e.g. PyTorch/XLA) lower `bf16_tensor < python_scalar` by
  // upcasting the tensor to a wider type and emitting the compare in the
  // wider type. This drops PyTorch eager semantics (which compares in bf16)
  // and flips the result at bf16-rounding boundaries. The conversion should
  // peel the artificial upcast and narrow the constant.
  func.func public @test_lt_bf16_through_f64_upcast(%arg0: tensor<1x2000x4xbf16>) -> tensor<1x2000x4xi1> {
    // CHECK-LABEL: func.func public @test_lt_bf16_through_f64_upcast
    // CHECK-NOT: ttir.typecast
    // CHECK-NOT: tensor<{{.*}}xf64>
    // CHECK: "ttir.constant"() <{value = dense<{{.*}}> : tensor<1x2000x4xbf16>}>
    // CHECK: = "ttir.lt"(%arg0, %{{[^)]+}})
    // CHECK-SAME: (tensor<1x2000x4xbf16>, tensor<1x2000x4xbf16>) -> tensor<1x2000x4xi1>
    %cst = stablehlo.constant dense<0.99> : tensor<1x2000x4xf64>
    %0 = stablehlo.convert %arg0 : (tensor<1x2000x4xbf16>) -> tensor<1x2000x4xf64>
    %1 = stablehlo.compare  LT, %0, %cst : (tensor<1x2000x4xf64>, tensor<1x2000x4xf64>) -> tensor<1x2000x4xi1>
    return %1 : tensor<1x2000x4xi1>
  }
}

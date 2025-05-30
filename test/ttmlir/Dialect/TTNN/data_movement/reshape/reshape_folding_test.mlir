// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s| FileCheck %s

module @reshape_test {
  // Test folding of "ttir.reshape" when called with identical shapes.
  func.func @main(%arg0: tensor<1xi32>) -> (tensor<1xi32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1xi32>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32]}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // CHECK-NOT: = "ttnn.reshape"
    return %1 : tensor<1xi32>
    // CHECK: return %arg0 : tensor<1xsi32, #{{.*}}>
  }
  // Test folding of "ttir.reshape" when called after a constant op.
  func.func @test_constant_reshape() -> tensor<1x2x1x1xf32> {
    %1 = "ttir.constant"() <{value = dense<[1.01, 2.02]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %2 = ttir.empty() : tensor<1x2x1x1xf32>
    // CHECK-NOT: = "ttnn.reshape"
    %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 2 : i32, 1 : i32, 1 : i32]}> : (tensor<2xf32>, tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
    // CHECK: return %0 : tensor<1x2x1x1xf32, #{{.*}}>
    return %3 : tensor<1x2x1x1xf32>
  }
}

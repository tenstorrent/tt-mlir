// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s| FileCheck %s

module @reshape_test {
  // Test folding of "ttir.reshape" when called with identical shapes.
  func.func @main(%arg0: tensor<1xi32>) -> (tensor<1xi32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1xi32>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32]}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // CHECK-NOT: = "ttnn.reshape"
    return %1 : tensor<1xi32>
    // CHECK: return %arg0 : tensor<1xi32, #{{.*}}>
  }
}

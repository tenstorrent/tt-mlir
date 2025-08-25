// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_dynamic_slice attributes {} {
    func.func @dynamic_slice(%arg0: tensor<32x64xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<8x8xf32> {jax.result_info = "result"}) {
    // CHECK: = "ttir.concat"
    // CHECK: = "ttir.constant"
    // CHECK: = "ttir.add"
    // CHECK: = ttir.empty
    // CHECK: = "ttir.slice_dynamic"
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [8, 8] : (tensor<32x64xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

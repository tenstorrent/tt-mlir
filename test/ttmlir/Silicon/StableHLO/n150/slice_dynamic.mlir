// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s -o %t.mlir --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%"
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// UNSUPPORTED: true
module @mod_slice_dynamic attributes {} {
    func.func @dynamic_slice(%arg0: tensor<32x64xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<8x8xf32> {jax.result_info = "result"}) {
    // CHECK: = "ttnn.reshape"
    // CHECK: = "ttnn.reshape"
    // CHECK: = "ttnn.concat"
    // CHECK: = "ttnn.add"
    // CHECK: = "ttnn.slice_dynamic"
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [8, 8] : (tensor<32x64xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

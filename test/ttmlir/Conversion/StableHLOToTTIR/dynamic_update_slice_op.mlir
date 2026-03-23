// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_dynamic_update_slice attributes {} {
  func.func @dynamic_update_slice(%base: tensor<1x197x768xf32>, %update: tensor<1x1x768xf32>, %idx0: tensor<i32>, %idx1: tensor<i32>, %idx2: tensor<i32>) -> tensor<1x197x768xf32> {
    // CHECK: = "ttir.concat"
    // CHECK: = "ttir.slice_write"
    %0 = stablehlo.dynamic_update_slice %base, %update, %idx0, %idx1, %idx2
      : (tensor<1x197x768xf32>, tensor<1x1x768xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x197x768xf32>
    return %0 : tensor<1x197x768xf32>
  }
}

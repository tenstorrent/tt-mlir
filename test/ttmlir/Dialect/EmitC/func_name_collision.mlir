// RUN: ttmlir-opt -ttir-to-emitc-pipeline="system-desc-path=%system_desc_path%" %s | FileCheck %s

// This test asserts that the function names in the TTIR module don't
// conflict with the function names that are introduced as a part of the pipeline.
module {
  func.func @main(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}

// CHECK-LABEL: func.func @_main

// CHECK-LABEL: func.func @create_inputs_for__main

// CHECK-LABEL: func.func @main
// CHECK: call @create_inputs_for__main
// CHECK: call @_main

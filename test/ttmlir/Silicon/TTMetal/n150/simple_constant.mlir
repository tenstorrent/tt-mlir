// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir

module {
  // CHECK-LABEL: func.func public @add5
  func.func public @add5(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK: memref.get_global
    // CHECK: "ttmetal.enqueue_write_buffer"
    // CHECK: "ttmetal.enqueue_write_buffer"
    // CHECK: memref.alloc
    // CHECK-NOT: #ttcore.host_layout
    // CHECK: "ttmetal.enqueue_read_buffer"
    %0 = "ttir.constant"() <{value = dense<5.0> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
    %1 = ttir.empty() : tensor<32x32xf32>
    %2 = "ttir.add"(%arg0, %0, %1) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %2 : tensor<32x32xf32>
  }

  // CHECK-LABEL: func.func public @add5_unaligned
  func.func public @add5_unaligned(%arg0: tensor<9x43x7xf32>) -> tensor<9x43x7xf32> {
    // CHECK: memref.get_global
    // CHECK: memref.alloc
    // CHECK-SAME: #ttcore.host_layout
    // CHECK: memref.copy
    // CHECK: "ttmetal.enqueue_write_buffer"
    // CHECK: memref.alloc
    // CHECK-SAME: #ttcore.host_layout
    // CHECK: memref.copy
    // CHECK: "ttmetal.enqueue_write_buffer"
    // CHECK: memref.alloc
    // CHECK-NOT: #ttcore.host_layout
    // CHECK: memref.alloc
    // CHECK-SAME: #ttcore.host_layout
    // CHECK: "ttmetal.enqueue_read_buffer"
    // CHECK: memref.copy
    %0 = "ttir.constant"() <{value = dense<5.0> : tensor<9x43x7xf32>}> : () -> tensor<9x43x7xf32>
    %1 = ttir.empty() : tensor<9x43x7xf32>
    %2 = "ttir.add"(%arg0, %0, %1) : (tensor<9x43x7xf32>, tensor<9x43x7xf32>, tensor<9x43x7xf32>) -> tensor<9x43x7xf32>
    return %2 : tensor<9x43x7xf32>
  }
}

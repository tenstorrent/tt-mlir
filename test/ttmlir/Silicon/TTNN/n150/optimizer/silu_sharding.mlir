// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module @"silu_sharding_test" {
  func.func @main(%arg0: tensor<1x256xf32>, %arg1: tensor<1x256xf32>) -> tensor<1x256xf32> {
    // CHECK-DAG: #[[LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}<width_sharded>>
    %0 = ttir.empty() : tensor<1x256xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x256xf32, #[[LAYOUT]]>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
    %2 = ttir.empty() : tensor<1x256xf32>
    // CHECK: %{{.*}} = "ttnn.silu"{{.*}} -> tensor<1x256xf32, #[[LAYOUT]]>
    %3 = "ttir.silu"(%1, %2) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
    return %3 : tensor<1x256xf32>
  }
}

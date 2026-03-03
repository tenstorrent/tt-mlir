// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false" -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
    // CHECK-DAG: #{{.*}} = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
    // CHECK-DAG: #{{.*}} = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

  // CHECK: ttcore.device_module {
  // CHECK: builtin.module attributes {{.*}} {
  // CHECK: func.func @forward
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: %{{.*}} = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %1 = "ttir.multiply"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.from_device"(%{{.*}}) : (tensor<[[DIMS:.*]], #{{.*}}>) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = call @cpu_hoisted_ttir_add_{{.*}}(%{{.*}}, %{{.*}})
    %3 = "ttir.add"(%arg0, %1) {ttir.should_hoist} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %{{.*}} = call @cpu_hoisted_ttir_add_{{.*}}(%{{.*}}, %{{.*}})
    %5 = "ttir.add"(%arg0, %3) {ttir.should_hoist} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.to_device"(%{{.*}}, %{{.*}}) <{memory_config = {{.*}}}> : (tensor<[[DIMS:.*]], #{{.*}}>, !ttnn.device) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.to_layout"(%{{.*}}) <{layout = #ttnn.layout<{{.*}}>}> : (tensor<[[DIMS:.*]], #{{.*}}>) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.to_device"(%{{.*}}, %{{.*}}) <{memory_config = {{.*}}}> : (tensor<[[DIMS:.*]], #{{.*}}>, !ttnn.device) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.to_layout"(%{{.*}}) <{layout = #ttnn.layout<{{.*}}>}> : (tensor<[[DIMS:.*]], #{{.*}}>) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %7 = "ttir.multiply"(%3, %5) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %7 : tensor<64x128xf32>
  }

  // Should have only ONE CPU-hoisted function since all add operations have
  // the same signature after type conversion.
  // CHECK-COUNT-1: func.func private @cpu_hoisted_ttir_add_{{.*}}

  // CHECK: ttcore.cpu_module {
  // CHECK: builtin.module {
  // CHECK-COUNT-1: llvm.func @cpu_hoisted_ttir_add_{{[^_]*}}(
  // CHECK: llvm.func @cpu_hoisted_ttir_add_{{.*}}_helper(%arg0: !llvm.ptr)
}

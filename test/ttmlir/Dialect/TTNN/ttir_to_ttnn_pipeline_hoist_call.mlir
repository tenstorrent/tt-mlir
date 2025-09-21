// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false" -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
    // CHECK-DAG: #{{.*}} = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
    // CHECK-DAG: #{{.*}} = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

  // CHECK: ttcore.device_module {
  // CHECK: builtin.module attributes {{.*}} {
  // CHECK: func.func @forward
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.ones"
    %2 = "ttir.ones"() <{shape = array<i32:64, 128>, dtype = bf16}> : () -> tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.from_device"(%{{.*}}) : (tensor<[[DIMS:.*]], #{{.*}}>) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = call @hoisted_ttir_add_64x128_64x128_64x128_func_decl(%{{.*}}, %{{.*}}, %{{.*}})
    %3 = "ttir.add"(%arg0, %1, %2) {ttir.should_hoist} : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.zeros"
    %4 = ttir.empty() : tensor<64x128xf32>
    // CHECK: %{{.*}} = call @hoisted_ttir_add_64x128_64x128_64x128_func_decl(%{{.*}}, %{{.*}}, %{{.*}})
    %5 = "ttir.add"(%arg0, %3, %4) {ttir.should_hoist} : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %6 = ttir.empty() : tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.to_layout"(%{{.*}}) <{layout = #ttnn.layout<{{.*}}>}> : (tensor<[[DIMS:.*]], #{{.*}}>) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.to_device"(%{{.*}}, %{{.*}}) <{memory_config = {{.*}}}> : (tensor<[[DIMS:.*]], #{{.*}}>, !ttnn.device) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.to_layout"(%{{.*}}) <{layout = #ttnn.layout<{{.*}}>}> : (tensor<[[DIMS:.*]], #{{.*}}>) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.to_device"(%{{.*}}, %{{.*}}) <{memory_config = {{.*}}}> : (tensor<[[DIMS:.*]], #{{.*}}>, !ttnn.device) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %7 = "ttir.multiply"(%3, %5, %6) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %7 : tensor<64x128xf32>
  }
  // CHECK: func.func private @hoisted_ttir_add_64x128_64x128_64x128_func_decl
  // CHECK: ttcore.cpu_module {
  // CHECK: builtin.module {
  // CHECK: llvm.func @hoisted_ttir_add_64x128_64x128_64x128_func
  // CHECK: llvm.func @hoisted_ttir_add_64x128_64x128_64x128_func_helper(%arg0: !llvm.ptr)
}

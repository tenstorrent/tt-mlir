// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
    // CHECK-DAG: #{{.*}} = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
    // CHECK-DAG: #{{.*}} = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: %{{.*}} = "ttnn.empty"(%{{.*}})
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.multiply"(%{{.*}}, %{{.*}}, %{{.*}})
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.ones"
    %2 = "ttir.ones"() <{shape = array<i32:64, 128>}> : () -> tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.from_device"(%{{.*}}) : (tensor<[[DIMS:.*]], #{{.*}}>) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = call @hoisted_func_decl(%{{.*}}, %{{.*}}, %{{.*}})
    %3 = call @hoisted_func_decl(%arg0, %1, %2) {ttir.hoisted_call} : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.zeros"
    %4 = "ttir.zeros"() <{shape = array<i32:64, 128>}> : () -> tensor<64x128xf32>
    %5 = call @hoisted_func_decl(%arg0, %3, %4) {ttir.hoisted_call} : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.empty"(%{{.*}})
    %6 = tensor.empty() : tensor<64x128xf32>
    // CHECK: %{{.*}} = "ttnn.to_layout"(%{{.*}}) <{layout = #ttnn.layout<{{.*}}>}> : (tensor<[[DIMS:.*]], #{{.*}}>) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.to_device"(%{{.*}}, %{{.*}}) <{memory_config = {{.*}}}> : (tensor<[[DIMS:.*]], #{{.*}}>, !tt.device<#{{.*}}>) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.to_layout"(%{{.*}}) <{layout = #ttnn.layout<{{.*}}>}> : (tensor<[[DIMS:.*]], #{{.*}}>) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.to_device"(%{{.*}}, %{{.*}}) <{memory_config = {{.*}}}> : (tensor<[[DIMS:.*]], #{{.*}}>, !tt.device<#{{.*}}>) -> tensor<[[DIMS]], #{{.*}}>
    // CHECK: %{{.*}} = "ttnn.multiply"(%{{.*}}, %{{.*}}, %{{.*}})
    %7 = "ttir.multiply"(%3, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %7 : tensor<64x128xf32>
  }
  func.func private @hoisted_func_decl(tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
}

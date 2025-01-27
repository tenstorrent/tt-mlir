// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: #{{.*}} = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    // CHECK: %{{.*}} = "ttnn.from_device"(%{{.*}}) : (tensor<64x128xf32, #[[LAYOUT:.*]]>) -> tensor<64x128xf32, #[[LAYOUT]]>
    // CHECK: %{{.*}} = "ttnn.from_device"(%{{.*}}) : (tensor<64x128xf32, #[[LAYOUT:.*]]>) -> tensor<64x128xf32, #[[LAYOUT]]>
    // CHECK: %{{.*}} = "ttnn.from_device"(%{{.*}}) : (tensor<64x128xf32, #[[LAYOUT:.*]]>) -> tensor<64x128xf32, #[[LAYOUT]]>
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: %[[C:.*]] = call @hoisted_func_decl(%{{.*}}, %{{.*}}, %{{.*}}) : (tensor<64x128xf32, #{{.*}}>, tensor<64x128xf32, #{{.*}}>, tensor<64x128xf32, #{{.*}}>) -> tensor<64x128xf32, #{{.*}}>
    %1 = call @hoisted_func_decl(%arg0, %arg1, %0) {hoisted_call} : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK : %{{.*}} = "ttnn.to_device"(%{{.*}}, %{{.*}}) :(tensor<64x128xf32, #[[LAYOUT:.*]]>, !tt.device<#[[LAYOUT]]>) -> tensor<64x128xf32, #[[LAYOUT]]>
    return %1 : tensor<64x128xf32>
  }
  func.func private @hoisted_func_decl(tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
}

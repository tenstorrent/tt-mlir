// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
    // CHECK-DAG: #[[dram_layout:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[system_layout_f32:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>

  // CHECK: tt.device_module {
  // CHECK: builtin.module attributes {{.*}} {
  // CHECK: func.func @forward
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %0 = ttir.empty() : tensor<64x128xbf16>
    // CHECK: %{{.*}} = "ttnn.multiply"({{.*}}, {{.*}}) : (tensor<64x128xbf16, #[[dram_layout]]>, tensor<64x128xbf16, #[[dram_layout]]>) -> tensor<64x128xbf16, #[[dram_layout]]>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: %{{.*}} = "ttnn.ones"
    // CHECK-SAME: -> tensor<64x128xbf16, #ttnn_layout>
    %2 = "ttir.ones"() <{shape = array<i32:64, 128>}> : () -> tensor<64x128xbf16>
    // CHECK: %{{.*}} = call @hoisted_ttir_add_64x128_64x128_64x128_func_decl({{.*}}, {{.*}}, {{.*}}) : (tensor<64x128xf32, #[[system_layout_f32]]>, tensor<64x128xf32, #[[system_layout_f32]]>, tensor<64x128xf32, #[[system_layout_f32]]>) -> tensor<64x128xf32, #[[system_layout_f32]]>
    %3 = "ttir.add"(%arg0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> {should_hoist} : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: %{{.*}} = "ttnn.construct_tensor"
    %4 = ttir.empty() : tensor<64x128xbf16>
    // CHECK: %{{.*}} = call @hoisted_ttir_add_64x128_64x128_64x128_func_decl(%{{.*}}, %{{.*}}, %{{.*}}) : (tensor<64x128xf32, #[[system_layout_f32]]>, tensor<64x128xf32, #[[system_layout_f32]]>, tensor<64x128xf32, #[[system_layout_f32]]>) -> tensor<64x128xf32, #[[system_layout_f32]]>
    %5 = "ttir.add"(%arg0, %3, %4) <{operandSegmentSizes = array<i32: 2, 1>}> {should_hoist} : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %6 = ttir.empty() : tensor<64x128xbf16>
    // CHECK: %{{.*}} = "ttnn.multiply"(%{{.*}}, %{{.*}}) : (tensor<64x128xbf16, #[[dram_layout]]>, tensor<64x128xbf16, #[[dram_layout]]>) -> tensor<64x128xbf16, #[[dram_layout]]>
    %7 = "ttir.multiply"(%3, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %7 : tensor<64x128xbf16>
  }
  // CHECK: func.func private @hoisted_ttir_add_64x128_64x128_64x128_func_decl
  // CHECK: tt.cpu_module {
  // CHECK: builtin.module {
  // CHECK: llvm.func @hoisted_ttir_add_64x128_64x128_64x128_func
  // CHECK: llvm.func @hoisted_ttir_add_64x128_64x128_64x128_func_helper(%arg0: !llvm.ptr)
}

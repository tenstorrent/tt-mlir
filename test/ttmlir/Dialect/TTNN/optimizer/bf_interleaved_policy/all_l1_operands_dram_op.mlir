// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=BFInterleaved" %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<6144x1024xbf16>, %arg1: tensor<1024x6144xbf16>) -> tensor<6144x6144xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK-DAG: #[[LAYOUT_5:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<24x4x!tt.tile<32x32, bf16>, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_6:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<4x24x!tt.tile<32x32, bf16>, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_7:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<24x24x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    %0 = tensor.empty() : tensor<6144x1024xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<6144x1024xbf16, #[[LAYOUT_5]]>
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<6144x1024xbf16>, tensor<6144x1024xbf16>) -> tensor<6144x1024xbf16>
    %2 = tensor.empty() : tensor<1024x6144xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<1024x6144xbf16, #[[LAYOUT_6]]>
    %3 = "ttir.relu"(%arg1, %2) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x6144xbf16>, tensor<1024x6144xbf16>) -> tensor<1024x6144xbf16>
    %4 = tensor.empty() : tensor<6144x6144xbf16>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<6144x6144xbf16, #[[LAYOUT_7]]>
    %5 = "ttir.matmul"(%1, %3, %4) : (tensor<6144x1024xbf16>, tensor<1024x6144xbf16>, tensor<6144x6144xbf16>) -> tensor<6144x6144xbf16>
    return %5 : tensor<6144x6144xbf16>
  }
}

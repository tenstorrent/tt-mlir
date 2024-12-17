// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=GreedyL1Interleaved" %s | FileCheck %s
//
//       A     B
//        \   /
//          C
//          |
//          D
//
//  (A + B + C > L1) AND (A + C < A + B) AND (B + C < A + B) AND (A + B <= L1)
//      =>
//  DRAM: C; L1: AB
//
module attributes {} {
  func.func @forward(%arg0: tensor<2048x8192xbf16>, %arg1: tensor<2048x8192xbf16>, %arg2: tensor<8192x2048xbf16>, %arg3: tensor<8192x2048xbf16>) -> tensor<2048x2048xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK-DAG: #[[LAYOUT_4:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<8x32x!tt.tile<32x32, bf16>, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_6:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<32x8x!tt.tile<32x32, bf16>, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_7:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<8x8x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    %0 = tensor.empty() : tensor<2048x8192xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<2048x8192xbf16, #[[LAYOUT_4]]>
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2048x8192xbf16>, tensor<2048x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<2048x8192xbf16>
    %2 = tensor.empty() : tensor<8192x2048xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<8192x2048xbf16, #[[LAYOUT_6]]>
    %3 = "ttir.add"(%arg2, %arg3, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<8192x2048xbf16>
    %4 = tensor.empty() : tensor<2048x2048xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<2048x2048xbf16, #[[LAYOUT_7]]>
    %5 = "ttir.matmul"(%1, %3, %4) : (tensor<2048x8192xbf16>, tensor<8192x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    return %5 : tensor<2048x2048xbf16>
  }
}

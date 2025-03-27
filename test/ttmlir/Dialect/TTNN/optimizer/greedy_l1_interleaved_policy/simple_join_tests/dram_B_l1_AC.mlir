// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=GreedyL1Interleaved" %s | FileCheck %s
//
//       A     B
//        \   /
//          C
//          |
//          D
//
//  (A + B + C > L1) AND (B + C < A + C) AND (A + B < A + C) AND (A + C <= L1)
//      =>
//  DRAM: B; L1: AC
//
module attributes {} {
  func.func @forward(%arg0: tensor<8192x2048xbf16>, %arg1: tensor<8192x2048xbf16>, %arg2: tensor<2048x2048xbf16>, %arg3: tensor<2048x2048xbf16>) -> tensor<8192x2048xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK-DAG: #[[LAYOUT_3:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<8x8x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_5:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x256x!tt.tile<32x32, bf16>, #l1_>, <interleaved>>
    %0 = ttir.empty() : tensor<8192x2048xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<8192x2048xbf16, #[[LAYOUT_5]]>
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<8192x2048xbf16>
    %2 = ttir.empty() : tensor<2048x2048xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<2048x2048xbf16, #[[LAYOUT_3]]>
    %3 = "ttir.add"(%arg2, %arg3, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %4 = ttir.empty() : tensor<8192x2048xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<8192x2048xbf16, #[[LAYOUT_5]]>
    %5 = "ttir.matmul"(%1, %3, %4) : (tensor<8192x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<8192x2048xbf16>
    return %5 : tensor<8192x2048xbf16>
  }
}

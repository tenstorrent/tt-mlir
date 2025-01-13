// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=BFInterleaved" %s | FileCheck %s
//
//         A
//         |
//         B
//       /   \
//      C     D
//      |     |
//      E     |
//       \   /
//         F
//         |
//         G
//
// There is not enough L1 memory to schedule this fork-join even if we allocate
// the output tensor of the op B once becuase the output tensor of the op C is
// too large to fit in L1 on its own.
//
module attributes {} {
  func.func @forward(%arg0: tensor<4096x5120xbf16>, %arg1: tensor<5120x9216xbf16>, %arg2: tensor<9216x1024xbf16>, %arg3: tensor<5120x1024xbf16>) -> tensor<4096x1024xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_9:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x320x!tt.tile<32x32, bf16>, #l1_>, <interleaved>>
    // CHECK: #[[LAYOUT_10:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<16x36x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK: #[[LAYOUT_11:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x64x!tt.tile<32x32, bf16>, #l1_>, <interleaved>>
    %0 = tensor.empty() : tensor<4096x5120xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<4096x5120xbf16, #[[LAYOUT_9]]>
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<4096x5120xbf16>, tensor<4096x5120xbf16>) -> tensor<4096x5120xbf16>
    %2 = tensor.empty() : tensor<4096x9216xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<4096x9216xbf16, #[[LAYOUT_10]]>
    %3 = "ttir.matmul"(%1, %arg1, %2) : (tensor<4096x5120xbf16>, tensor<5120x9216xbf16>, tensor<4096x9216xbf16>) -> tensor<4096x9216xbf16>
    %4 = tensor.empty() : tensor<4096x1024xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_11]]>
    %5 = "ttir.matmul"(%3, %arg2, %4) : (tensor<4096x9216xbf16>, tensor<9216x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    %6 = tensor.empty() : tensor<4096x1024xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_11]]>
    %7 = "ttir.matmul"(%1, %arg3, %6) : (tensor<4096x5120xbf16>, tensor<5120x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    %8 = tensor.empty() : tensor<4096x1024xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_11]]>
    %9 = "ttir.add"(%5, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<4096x1024xbf16>, tensor<4096x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    return %9 : tensor<4096x1024xbf16>
  }
}

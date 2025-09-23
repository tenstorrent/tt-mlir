// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=BFInterleaved tensor-l1-usage-cap=0.75" -o %t %s
// RUN: FileCheck %s --input-file=%t
// XFAIL: *
// OpValidation pass finds matmuls OOM. Needs to be tested and fixed.
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
    // CHECK: #[[LAYOUT_4:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x64x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
    // CHECK: #[[LAYOUT_5:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x320x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
    // CHECK: #[[LAYOUT_6:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x288x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    %0 = ttir.empty() : tensor<4096x5120xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<4096x5120xbf16, #[[LAYOUT_5]]>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<4096x5120xbf16>, tensor<4096x5120xbf16>) -> tensor<4096x5120xbf16>
    %2 = ttir.empty() : tensor<4096x9216xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<4096x9216xbf16, #[[LAYOUT_6]]>
    %3 = "ttir.matmul"(%1, %arg1, %2) : (tensor<4096x5120xbf16>, tensor<5120x9216xbf16>, tensor<4096x9216xbf16>) -> tensor<4096x9216xbf16>
    %4 = ttir.empty() : tensor<4096x1024xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_4]]>
    %5 = "ttir.matmul"(%3, %arg2, %4) : (tensor<4096x9216xbf16>, tensor<9216x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    %6 = ttir.empty() : tensor<4096x1024xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_4]]>
    %7 = "ttir.matmul"(%1, %arg3, %6) : (tensor<4096x5120xbf16>, tensor<5120x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    %8 = ttir.empty() : tensor<4096x1024xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_4]]>
    %9 = "ttir.add"(%5, %7, %8) : (tensor<4096x1024xbf16>, tensor<4096x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    return %9 : tensor<4096x1024xbf16>
  }
}

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
//       \   /
//         E
//         |
//         F
//
// There is enough L1 memory to schedule this fork-join but only if we allocate
// the output tensor of the op B once.
//
module attributes {} {
  func.func @forward(%arg0: tensor<4096x5120xbf16>, %arg1: tensor<5120x1024xbf16>, %arg2: tensor<5120x1024xbf16>) -> tensor<4096x1024xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_2:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x64x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
    // CHECK: #[[LAYOUT_3:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x320x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
    %0 = ttir.empty() : tensor<4096x5120xbf16>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<4096x5120xbf16, #[[LAYOUT_3]]>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<4096x5120xbf16>, tensor<4096x5120xbf16>) -> tensor<4096x5120xbf16>
    %2 = ttir.empty() : tensor<4096x1024xbf16>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_2]]>
    %3 = "ttir.matmul"(%1, %arg1, %2) : (tensor<4096x5120xbf16>, tensor<5120x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    %4 = ttir.empty() : tensor<4096x1024xbf16>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_2]]>
    %5 = "ttir.matmul"(%1, %arg2, %4) : (tensor<4096x5120xbf16>, tensor<5120x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    %6 = ttir.empty() : tensor<4096x1024xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_2]]>
    %7 = "ttir.add"(%3, %5, %6) : (tensor<4096x1024xbf16>, tensor<4096x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    return %7 : tensor<4096x1024xbf16>
  }
}

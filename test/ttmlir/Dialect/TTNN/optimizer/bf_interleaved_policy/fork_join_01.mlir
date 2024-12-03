// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=BFInterleaved" %s | FileCheck %s
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
// There is enough L1 memory to schedule this simple fork-join without any DRAM spill.
//
module attributes {} {
  func.func @forward(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_2:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!tt.tile<32x32, f32>, #l1_>, <interleaved>>
    %0 = tensor.empty() : tensor<32x32xf32>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<32x32xf32, #[[LAYOUT_2]]>
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %2 = tensor.empty() : tensor<32x32xf32>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<32x32xf32, #[[LAYOUT_2]]>
    %3 = "ttir.relu"(%1, %2) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = tensor.empty() : tensor<32x32xf32>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<32x32xf32, #[[LAYOUT_2]]>
    %5 = "ttir.relu"(%1, %4) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %6 = tensor.empty() : tensor<32x32xf32>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x32xf32, #[[LAYOUT_2]]>
    %7 = "ttir.add"(%3, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %7 : tensor<32x32xf32>
  }
}

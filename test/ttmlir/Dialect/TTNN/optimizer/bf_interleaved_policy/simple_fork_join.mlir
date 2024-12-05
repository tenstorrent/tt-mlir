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
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_2:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<4x4xbf16, #l1_>, interleaved>
    %0 = tensor.empty() : tensor<32x32xbf16>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<32x32xbf16, #[[LAYOUT_2]]>
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = tensor.empty() : tensor<32x32xbf16>
    // CHECK: %{{.*}} = "ttnn.abs"{{.*}} -> tensor<32x32xbf16, #[[LAYOUT_2]]>
    %3 = "ttir.abs"(%1, %2) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = tensor.empty() : tensor<32x32xbf16>
    // CHECK: %{{.*}} = "ttnn.gelu"{{.*}} -> tensor<32x32xbf16, #[[LAYOUT_2]]>
    %5 = "ttir.gelu"(%1, %4) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %6 = tensor.empty() : tensor<32x32xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x32xbf16, #[[LAYOUT_2]]>
    %7 = "ttir.add"(%3, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %7 : tensor<32x32xbf16>
  }
}

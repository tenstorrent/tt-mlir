// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=L1Interleaved" %s | FileCheck %s
//
//      B     C
//       \   /
//   A     D
//    \   /
//      E
//      |
//      F
//
//  (A + B + C + D > L1) AND (B + C + D <= L1) AND (A + D + E <= L1)
//      =>
//  DRAM: None; L1: ABCDE
//  Precedence of E: [D, A]
//
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<2048x5120xbf16>, %arg1: tensor<2048x5120xbf16>, %arg2: tensor<2048x5120xbf16>) -> tensor<2048x5120xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_2:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <{{.*}}>, memref<256x640xbf16, #l1_>, interleaved>
    %0 = tensor.empty() : tensor<2048x5120xbf16>
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<2048x5120xbf16>, tensor<2048x5120xbf16>) -> tensor<2048x5120xbf16>
    %2 = tensor.empty() : tensor<2048x5120xbf16>
    %3 = "ttir.add"(%arg1, %arg2, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<2048x5120xbf16>, tensor<2048x5120xbf16>, tensor<2048x5120xbf16>) -> tensor<2048x5120xbf16>
    %4 = tensor.empty() : tensor<2048x5120xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<2048x5120xbf16, #[[LAYOUT_2]]>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<2048x5120xbf16, #[[LAYOUT_2]]>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<2048x5120xbf16, #[[LAYOUT_2]]>
    %5 = "ttir.add"(%1, %3, %4) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<2048x5120xbf16>, tensor<2048x5120xbf16>, tensor<2048x5120xbf16>) -> tensor<2048x5120xbf16>
    return %5 : tensor<2048x5120xbf16>
  }
}

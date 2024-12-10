// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=L1Interleaved" %s | FileCheck %s
//
//       A     B
//        \   /
//          C
//          |
//          D
//
//  (A + B + C > L1) AND (A + C < B + C) AND (A + B < B + C) AND (B + C <= L1)
//      =>
//  DRAM: A; L1: BC
//
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<2048x2048xbf16>, %arg1: tensor<2048x2048xbf16>, %arg2: tensor<2048x8192xbf16>, %arg3: tensor<2048x8192xbf16>) -> tensor<2048x8192xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK-DAG: #[[LAYOUT_3:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<8x8x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_5:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<8x32x!tt.tile<32x32, bf16>, #l1_>, <interleaved>>
    %0 = tensor.empty() : tensor<2048x2048xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<2048x2048xbf16, #[[LAYOUT_3]]>
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %2 = tensor.empty() : tensor<2048x8192xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<2048x8192xbf16, #[[LAYOUT_5]]>
    %3 = "ttir.add"(%arg2, %arg3, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<2048x8192xbf16>, tensor<2048x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<2048x8192xbf16>
    %4 = tensor.empty() : tensor<2048x8192xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<2048x8192xbf16, #[[LAYOUT_5]]>
    %5 = "ttir.matmul"(%1, %3, %4) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<2048x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<2048x8192xbf16>
    return %5 : tensor<2048x8192xbf16>
  }
}

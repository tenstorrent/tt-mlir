// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=L1Interleaved" %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>, %arg2: tensor<64x96xbf16>, %arg3: tensor<96x32xbf16>, %arg4: tensor<64x32xbf16>) -> tensor<64x32xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_7:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <{{.*}}>, memref<{{.*}}, #l1_>, <interleaved>>
    // CHECK: #[[LAYOUT_10:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <{{.*}}>, memref<{{.*}}, #l1_>, <interleaved>>
    %0 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_7]]>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %2 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_7]]>
    %3 = "ttir.add"(%1, %arg2, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x96xbf16>, tensor<64x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %4 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_7]]>
    %5 = "ttir.relu"(%3, %4) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %6 = tensor.empty() : tensor<64x32xbf16>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_10]]>
    %7 = "ttir.matmul"(%5, %arg3, %6) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x96xbf16>, tensor<96x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    %8 = tensor.empty() : tensor<64x32xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_10]]>
    %9 = "ttir.add"(%7, %arg4, %8) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x32xbf16>, tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    %10 = tensor.empty() : tensor<64x32xbf16>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_10]]>
    %11 = "ttir.relu"(%9, %10) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %11 : tensor<64x32xbf16>
  }
}

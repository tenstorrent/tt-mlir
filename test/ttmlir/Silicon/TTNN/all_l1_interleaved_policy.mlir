// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=L1Interleaved" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>, %arg2: tensor<64x96xbf16>, %arg3: tensor<96x32xbf16>, %arg4: tensor<64x32xbf16>) -> tensor<64x32xbf16> {
    // CHECK: #[[L1_:.*]] = #tt.memory_space<l1>
    // CHECK: #[[LAYOUT_6:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<8x12xbf16, #l1_>, interleaved>
    // CHECK: #[[LAYOUT_7:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<8x4xbf16, #l1_>, interleaved>
    // CHECK: #[[LAYOUT_8:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<8x4xbf16, #dram>, interleaved>
    %0 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_6]]>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %2 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_6]]>
    %3 = "ttir.add"(%1, %arg2, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x96xbf16>, tensor<64x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %4 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_6]]>
    %5 = "ttir.relu"(%3, %4) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %6 = tensor.empty() : tensor<64x32xbf16>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_7]]>
    %7 = "ttir.matmul"(%5, %arg3, %6) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x96xbf16>, tensor<96x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    %8 = tensor.empty() : tensor<64x32xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_7]]>
    %9 = "ttir.add"(%7, %arg4, %8) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x32xbf16>, tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    %10 = tensor.empty() : tensor<64x32xbf16>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_8]]>
    %11 = "ttir.relu"(%9, %10) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %11 : tensor<64x32xbf16>
  }
}

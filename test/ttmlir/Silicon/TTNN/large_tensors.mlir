// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=L1Interleaved" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<8192x8192xbf16>, %arg1: tensor<8192x8192xbf16>, %arg2: tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16> {
    // CHECK: #[[LAYOUT_2:layout2]] = #tt.layout<{{.*}}, memref<{{.*}}, #dram>, {{.*}}>
    %0 = tensor.empty() : tensor<8192x8192xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<8192x8192xbf16, #[[LAYOUT_2]]>
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<8192x8192xbf16>, tensor<8192x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>
    %2 = tensor.empty() : tensor<8192x8192xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<8192x8192xbf16, #[[LAYOUT_2]]>
    %3 = "ttir.add"(%1, %arg2, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<8192x8192xbf16>, tensor<8192x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>
    %4 = tensor.empty() : tensor<8192x8192xbf16>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<8192x8192xbf16, #[[LAYOUT_2]]>
    %7 = "ttir.relu"(%3, %4) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<8192x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>
    return %7 : tensor<8192x8192xbf16>
  }
}

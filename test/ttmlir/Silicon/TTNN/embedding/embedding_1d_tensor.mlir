// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s  > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<32xbf16>, %arg1: tensor<512x128xbf16>) -> tensor<32x128xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<32x128xbf16>
    // CHECK: %[[C:.*]] = "ttnn.embedding"[[C:.*]]
    %1 = "ttir.embedding"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32xbf16>, tensor<512x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }
}

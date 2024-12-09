// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module {
  func.func @forward(%arg0: tensor<1x32x64x512xbf16>, %arg1: tensor<1x32x3x512xbf16>) -> tensor<1x32x64x512xbf16> {
    // CHECK: "ttnn.fill_cache"[[C:.*]]
    %1 = "ttir.fill_cache"(%arg0, %arg1) <{batch_offset = 0: i32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x32x64x512xbf16>, tensor<1x32x3x512xbf16>) -> tensor<1x32x64x512xbf16>
    %cst = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1x32x64x512xbf16>}> : () -> tensor<1x32x64x512xbf16>
    %addition_dps = tensor.empty() : tensor<1x32x64x512xbf16>
    %2 = "ttir.add"(%1, %cst, %addition_dps) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x32x64x512xbf16>, tensor<1x32x64x512xbf16>, tensor<1x32x64x512xbf16>) -> tensor<1x32x64x512xbf16>
    return %2 : tensor<1x32x64x512xbf16>
  }
}

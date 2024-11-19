// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @upsample_scale_unifrom(%arg0: tensor<4x32x64x3xbf16>) -> tensor<4x64x128x3xbf16> {
    %0 = tensor.empty() : tensor<4x64x128x3xbf16>
    // CHECK: "ttnn.upsample"
    // CHECK-SAME: tensor<4x32x64x3xbf16
    // CHECK-SAME: tensor<4x64x128x3xbf16
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 2 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<4x32x64x3xbf16>, tensor<4x64x128x3xbf16>) -> tensor<4x64x128x3xbf16>
    return %1 : tensor<4x64x128x3xbf16>
  }
}

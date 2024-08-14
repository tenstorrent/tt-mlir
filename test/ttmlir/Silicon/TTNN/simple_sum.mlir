// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-implicit-device --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s  > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x32xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<512x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.sum"[[C:.*]]
    %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-1: i32], keep_dim = true, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x32xbf16>) -> tensor<512x32xbf16>
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %1 : tensor<512x32xbf16>
  }
}

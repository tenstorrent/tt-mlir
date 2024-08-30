// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// UNSUPPORTED: true
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<1x32xf32>, %arg1: tensor<512x128xf32>) -> tensor<1x32x128xf32> {
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<1x32x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.embedding"[[C:.*]]
    %1 = "ttir.embedding"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x32xf32>, tensor<512x128xf32>, tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %1 : tensor<1x32x128xf32>
  }
}

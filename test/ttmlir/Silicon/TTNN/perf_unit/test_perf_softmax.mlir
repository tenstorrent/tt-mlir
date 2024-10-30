// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>

func.func @softmax(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
  %0 = tensor.empty() : tensor<512x1024xbf16>
  // CHECK: %[[C:.*]] = "ttnn.softmax"[[C:.*]]
  // Check for positive dimension attribute
  %1 = "ttir.softmax"(%arg0, %0) <{dimension = 1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
  %2 = tensor.empty() : tensor<512x1024xbf16>
  // CHECK: %[[C:.*]] = "ttnn.softmax"[[C:.*]]
  // Check for negative dimension attribute
  %3 = "ttir.softmax"(%1, %2) <{dimension = -1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
  return %3 : tensor<512x1024xbf16>
}

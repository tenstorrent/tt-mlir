// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>

func.func public @broadcast() -> (tensor<32xf32>) {
  %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %1 = tensor.empty() : tensor<32xf32>
  %2 = "ttir.broadcast"(%0, %1) <{dimension = [0], operand_constraints = [#tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>, #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>]}> : (tensor<1xf32>, tensor<32xf32>) -> tensor<32xf32>
  %3 = tensor.empty() : tensor<32xf32>
  %4 = "ttir.broadcast"(%2, %3) <{dimension = [0], operand_constraints = [#tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>, #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>]}> : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  // CHECK-NOT: %[[C:.*]] = "ttir.broadcast"[[C:.*]]
  return %4 : tensor<32xf32>
}

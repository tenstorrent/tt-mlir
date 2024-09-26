// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>

func.func public @broadcast(%arg0: tensor<32x32xf32>) -> (tensor<32x32xf32>) {
  %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %1 = tensor.empty() : tensor<32x1xf32>
  %2 = "ttir.max"(%arg0, %1) <{dim_arg = [1 : i32], keep_dim = true, operand_constraints = [#tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>, #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>]}> : (tensor<32x32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
  %3 = tensor.empty() : tensor<32xf32>
  %4 = "ttir.reshape"(%2, %3) <{operand_constraints = [#tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>, #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>], shape = [32 : i32]}> : (tensor<32x1xf32>, tensor<32xf32>) -> tensor<32xf32>
  %5 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %6 = tensor.empty() : tensor<32xf32>
  %7 = "ttir.broadcast"(%5, %6) <{dimension = [], operand_constraints = [#tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>, #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>]}> : (tensor<1xf32>, tensor<32xf32>) -> tensor<32xf32>
  %8 = tensor.empty() : tensor<32xf32>
  %9 = "ttir.maximum"(%7, %4, %8) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>, #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>, #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>]}> : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %10 = tensor.empty() : tensor<32x1xf32>
  %11 = "ttir.broadcast"(%9, %10) <{dimension = [0], operand_constraints = [#tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>, #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>]}> : (tensor<32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
  %12 = tensor.empty() : tensor<32x32xf32>
  %13 = "ttir.broadcast"(%11, %12) <{dimension = [0, 1], operand_constraints = [#tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>, #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>]}> : (tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NOT: %[[C:.*]] = "ttir.broadcast"[[C:.*]]
  return %13 : tensor<32x32xf32>
}

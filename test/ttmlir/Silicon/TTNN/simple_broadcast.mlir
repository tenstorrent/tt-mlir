// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device_tile = #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>

func.func public @main(%arg0: tensor<1xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<512x512xf32> {mhlo.layout_mode = "default"}) -> (tensor<512x512xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  // CHECK-NOT: %[[C:.*]] = "ttnn.broadcast"[[C:.*]]
  %0 = tensor.empty() : tensor<512x512xf32>
  %1 = "ttir.broadcast"(%arg0, %0) <{dimension = [1], operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<1xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %2 = tensor.empty() : tensor<512x512xf32>
  %3 = "ttir.maximum"(%1, %arg1, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  return %3 : tensor<512x512xf32>
}

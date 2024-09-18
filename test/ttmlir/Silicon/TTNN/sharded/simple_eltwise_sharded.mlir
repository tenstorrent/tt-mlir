// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=false" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#l1_block_sharded = #tt.operand_constraint<l1_block_sharded>
#l1_block_sharded_tile = #tt.operand_constraint<l1|block_sharded|tile>
#l1_height_sharded = #tt.operand_constraint<l1|height_sharded|scalar|tile>

func.func @subtract(%arg0: tensor<224x64xf32>, %arg1: tensor<224x64xf32>) -> tensor<224x64xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<224x64xf32>
  // CHECK: %[[C:.*]] = "ttnn.subtract"[[C:.*]]
  %1 = "ttir.subtract"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#l1_block_sharded_tile, #l1_block_sharded_tile, #l1_block_sharded_tile]}> : (tensor<224x64xf32>, tensor<224x64xf32>, tensor<224x64xf32>) -> tensor<224x64xf32>
  return %1 : tensor<224x64xf32>
}

/////////////////////////////////////////
// Unsupported eltwise ops with sharding
//  * Concat: Sharded concat requires ROW MAJOR layout
/////////////////////////////////////////

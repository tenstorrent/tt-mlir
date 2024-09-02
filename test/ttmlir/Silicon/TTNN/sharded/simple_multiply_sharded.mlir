// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#l1_block_sharded = #tt.operand_constraint<l1_block_sharded>
module attributes {} {
  func.func @forward(%arg0: tensor<256x512xbf16>, %arg1: tensor<256x512xbf16>) -> tensor<256x512xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<256x512xbf16>
    // CHECK: %[[C:.*]] = "ttnn.multiply"[[C:.*]]
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#l1_block_sharded, #l1_block_sharded, #l1_block_sharded]}> : (tensor<256x512xbf16>, tensor<256x512xbf16>, tensor<256x512xbf16>) -> tensor<256x512xbf16>
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %1 : tensor<256x512xbf16>
  }
}

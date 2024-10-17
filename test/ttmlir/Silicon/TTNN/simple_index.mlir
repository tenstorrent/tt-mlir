// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device_tile = #tt.operand_constraint<dram|l1|interleaved>
module attributes {} {
  func.func @forward(%arg0: tensor<4x32x32xbf16>) -> tensor<4x32x16xbf16> {
    %0 = tensor.empty() : tensor<4x32x16xbf16>
    // CHECK: %[[C:.*]] = "ttnn.slice"[[C:.*]]
    %1 = "ttir.index"(%arg0, %0) <{dim = 2: i32, begin = 0: i32, end = 32: i32, step = 2: i32, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<4x32x32xbf16>, tensor<4x32x16xbf16>) -> tensor<4x32x16xbf16>
    return %1 : tensor<4x32x16xbf16>
  }
}

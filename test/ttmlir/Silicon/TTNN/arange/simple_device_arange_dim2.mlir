// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// UNSUPPORTED: true
// https://github.com/tenstorrent/tt-mlir/issues/1448
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.arange"[[C:.*]]
    %0 = "ttir.arange"() <{start = 0: si64, end = 64: si64, step = 2: si64, arange_dimension = 2: i64}> : () -> tensor<1x1x32x128xbf16>
    %1 = tensor.empty() : tensor<1x1x32x128xbf16>
    %2 = "ttir.multiply"(%arg0, %0, %1) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    return %2 : tensor<1x1x32x128xbf16>
  }
}

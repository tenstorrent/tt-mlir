// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module {
  func.func @forward(%arg0: tensor<1x32x64x512xbf16>, %arg1: tensor<1x32x1x512xbf16>) -> tensor<1x32x64x512xbf16> {
    // CHECK: "ttnn.update_cache"[[C:.*]]
    %update_index = "ttir.constant"() <{value = dense<0> : tensor<1xui32>}> : () -> tensor<1xui32>
    %1 = "ttir.update_cache"(%arg0, %arg1, %update_index) <{batch_offset = 0: i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x1x512xbf16>, tensor<1xui32>) -> tensor<1x32x64x512xbf16>
    %cst = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1x32x64x512xbf16>}> : () -> tensor<1x32x64x512xbf16>
    %addition_dps = tensor.empty() : tensor<1x32x64x512xbf16>
    %2 = "ttir.add"(%1, %cst, %addition_dps) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x64x512xbf16>, tensor<1x32x64x512xbf16>, tensor<1x32x64x512xbf16>) -> tensor<1x32x64x512xbf16>
    return %2 : tensor<1x32x64x512xbf16>
  }
}

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @forward(%arg0: tensor<1x32x64x512xbf16>, %arg1: tensor<1x32x3x512xbf16>) -> tensor<1x32x64x512xbf16> {
    // CHECK: "ttnn.fill_cache"
    %1 = "ttir.fill_cache"(%arg0, %arg1) <{batch_offset = 0: i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x3x512xbf16>) -> tensor<1x32x64x512xbf16>
    %cst = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1x32x64x512xbf16>}> : () -> tensor<1x32x64x512xbf16>
    %addition_dps = ttir.empty() : tensor<1x32x64x512xbf16>
    %2 = "ttir.add"(%1, %cst, %addition_dps) : (tensor<1x32x64x512xbf16>, tensor<1x32x64x512xbf16>, tensor<1x32x64x512xbf16>) -> tensor<1x32x64x512xbf16>
    return %2 : tensor<1x32x64x512xbf16>
  }
}

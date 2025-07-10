// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module {
  func.func @concatenate_heads(%arg0: tensor<1x24x32x128xbf16>) -> tensor<1x32x3072xbf16> {
    %0 = ttir.empty() : tensor<1x32x3072xbf16>
    // CHECK: "ttnn.concatenate_heads"(%arg0)
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x24x32x128xbf16>, tensor<1x32x3072xbf16>) -> tensor<1x32x3072xbf16>
    return %1 : tensor<1x32x3072xbf16>
  }
}

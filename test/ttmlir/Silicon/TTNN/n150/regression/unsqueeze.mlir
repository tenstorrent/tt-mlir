// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module @Unsqueeze attributes {} {
  func.func @forward(%arg0: tensor<32xbf16> {ttir.name = "a"}) -> (tensor<1x32xbf16> {ttir.name = "Unsqueeze_393.output_unsqueeze_1214"}) {
    %0 = ttir.empty() : tensor<1x32xbf16>
    // CHECK: = "ttnn.reshape"
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = 0 : si32}> : (tensor<32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    return %1 : tensor<1x32xbf16>
  }
}

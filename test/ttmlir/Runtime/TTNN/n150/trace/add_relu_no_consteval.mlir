// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false enable-trace=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @add_with_relu(%arg0: tensor<512x512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<512x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<512x512xbf16> {
    // CHECK: ttnn.capture_or_execute_trace
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<512x512xbf16>, tensor<512x512xbf16>) -> tensor<512x512xbf16>
    %2 = "ttir.relu"(%1) : (tensor<512x512xbf16>) -> tensor<512x512xbf16>
    return %2 : tensor<512x512xbf16>
  }
}

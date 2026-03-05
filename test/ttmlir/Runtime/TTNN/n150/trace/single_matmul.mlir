// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false enable-trace=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @single_matmul(%arg0: tensor<256x256xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<256x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<256x256xbf16> {
    // CHECK: ttnn.capture_or_execute_trace
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
    return %0 : tensor<256x256xbf16>
  }
}

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=true enable-trace=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @matmul_with_div(%arg0: tensor<784x1096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<1096x784xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<784x784xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<784x784xbf16> {    // CHECK: ttcore.load_cached
    // CHECK: ttnn.capture_or_execute_trace
    %0 = ttir.empty() : tensor<784x784xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<784x1096xbf16>, tensor<1096x784xbf16>, tensor<784x784xbf16>) -> tensor<784x784xbf16>
    %2 = ttir.empty() : tensor<784x784xbf16>
    %3 = "ttir.div"(%1, %arg2, %2) : (tensor<784x784xbf16>, tensor<784x784xbf16>, tensor<784x784xbf16>) -> tensor<784x784xbf16>
    return %3 : tensor<784x784xbf16>
  }
}

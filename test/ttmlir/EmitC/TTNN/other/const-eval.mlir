// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=true" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir
// RUN: FileCheck %s --input-file %t.mlir

module {
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = ttcore.load_cached(@forward_const_eval_0, [%arg1, %arg2, %arg3])
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttnn.add"(%arg0, %arg1)
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2, %2)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3, %4)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %6 = ttir.empty() : tensor<32x32xbf16>
    %7 = "ttir.subtract"(%3, %5, %6) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %8 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%1, %7, %8) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }

  func.func @const_eval_no_input(%arg0: tensor<71x4x1xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<71x4x1xbf16> {
    // CHECK: = ttcore.load_cached(@const_eval_no_input_const_eval_0, [])
    %0 = "ttir.full"() <{fill_value = 32.0 : f32, shape = array<i32>}> : () -> tensor<bf16>
    %1 = ttir.empty() : tensor<1x1xbf16>
    %2 = "ttir.reshape"(%0, %1) <{shape = [1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    %3 = ttir.empty() : tensor<71x4xbf16>
    %4 = "ttir.repeat"(%2, %3) <{repeat_dimensions = array<i64: 71, 4>}> : (tensor<1x1xbf16>, tensor<71x4xbf16>) -> tensor<71x4xbf16>
    %5 = ttir.empty() : tensor<71x4x1xbf16>
    %6 = "ttir.reshape"(%4, %5) <{shape = [71 : i32, 4 : i32, 1 : i32]}> : (tensor<71x4xbf16>, tensor<71x4x1xbf16>) -> tensor<71x4x1xbf16>
    %7 = ttir.empty() : tensor<71x4x1xbf16>
    %8 = "ttir.multiply"(%6, %arg0, %7) : (tensor<71x4x1xbf16>, tensor<71x4x1xbf16>, tensor<71x4x1xbf16>) -> tensor<71x4x1xbf16>
    return %8 : tensor<71x4x1xbf16>
  }
}

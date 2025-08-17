// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=true" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir
// RUN: FileCheck %s --input-file %t.mlir

module {
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = ttcore.load_cached(@forward_const_eval_0, [%arg1, %arg2, %arg3])
    // CHECK: = "ttnn.add"(%arg0, %arg1)
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2)  : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3)  : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %7 = "ttir.subtract"(%3, %5) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%1, %7) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }
}

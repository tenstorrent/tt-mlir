// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=true" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @embedding(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1: tensor<512x128xbf16>) -> tensor<32x32x128xbf16> {
  %0 = ttir.empty() : tensor<32x32x128xbf16>
  %1 = "ttir.embedding"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<512x128xbf16>, tensor<32x32x128xbf16>) -> tensor<32x32x128xbf16>
  return %1 : tensor<32x32x128xbf16>
}

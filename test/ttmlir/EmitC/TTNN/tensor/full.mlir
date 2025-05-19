// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @full() -> tensor<32x32xbf16> {
  %0 = "ttir.constant"() <{value = dense<13.89> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

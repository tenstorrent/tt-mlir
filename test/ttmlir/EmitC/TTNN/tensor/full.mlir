// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline="target-dylib=true tuplify-input-if-empty=true" %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @full_float() -> tensor<64x128xbf16> {
  %0 = "ttir.full"() <{shape = array<i32: 64, 128>, fill_value = 3.0 : f32}> : () -> tensor<64x128xbf16>
  return %0 : tensor<64x128xbf16>
}

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-tuplify-tensors="tuplify-input-if-empty=true" --convert-ttnn-to-emitc -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @full_float() -> tensor<64x128xbf16> {
  %0 = "ttir.full"() <{shape = array<i32: 64, 128>, fill_value = 3.0 : f32}> : () -> tensor<64x128xbf16>
  return %0 : tensor<64x128xbf16>
}

func.func @full_int() -> tensor<64x128xi32> {
  %0 = "ttir.full"() <{shape = array<i32: 64, 128>, fill_value = 3 : i32}> : () -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}

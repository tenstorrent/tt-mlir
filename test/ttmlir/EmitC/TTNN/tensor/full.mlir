// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @full() -> tensor<4x12x15x31xf32> {
  %0 = "ttir.full"() <{shape = array<i32: 4, 12, 15, 31>, fill_value = 13.89 : f32}> : () -> tensor<4x12x15x31xf32>
  return %0 : tensor<4x12x15x31xf32>
}

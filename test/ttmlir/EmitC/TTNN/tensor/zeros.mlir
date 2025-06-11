// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors="tuplify-input-if-empty=true" --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @zeros() -> tensor<13x24x56x42xbf16> {
  %0 = "ttir.zeros"() <{shape = array<i32:13, 24, 56, 42>}> : () -> tensor<13x24x56x42xbf16>
  return %0 : tensor<13x24x56x42xbf16>
}

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline="tuplify-input-if-empty=true" -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @ones() -> tensor<13x24x56x42xbf16> {
  %0 = "ttir.ones"() <{shape = array<i32:13, 24, 56, 42>, dtype = bf16}> : () -> tensor<13x24x56x42xbf16>
  return %0 : tensor<13x24x56x42xbf16>
}

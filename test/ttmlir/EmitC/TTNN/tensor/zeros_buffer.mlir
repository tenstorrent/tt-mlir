// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path% enable-const-eval=false" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline="tuplify-input-if-empty=true" -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir
// RUN: FileCheck %s --input-file=%basename_t.cpp

// There is no ttnn::zeros_buffer in the TTNN API -- the op exists only to
// carry compile-time semantics. The emitted C++ must call ttnn::zeros.
// CHECK: ttnn::zeros
// CHECK-NOT: ttnn::zeros_buffer

func.func @zeros_buffer() -> tensor<13x24x56x42xbf16> {
  %0 = "ttir.zeros_buffer"() <{shape = array<i32:13, 24, 56, 42>}> : () -> tensor<13x24x56x42xbf16>
  return %0 : tensor<13x24x56x42xbf16>
}

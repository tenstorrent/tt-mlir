// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir
// UNSUPPORTED: true
// https://github.com/tenstorrent/tt-mlir/issues/4383

func.func @test_rand() -> tensor<32x32xbf16>{
  %0 = "ttir.rand"() <{dtype = bf16, size = [32 : i32, 32 : i32]}> : () -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

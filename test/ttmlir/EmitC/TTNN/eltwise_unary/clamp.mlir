// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_fb.mlir
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @clamp(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %0 = "ttir.clamp_scalar"(%arg0) <{max = 3.000000e+00 : f32, min = 2.000000e+00 : f32}> : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %0 : tensor<64x128xbf16>
}

func.func @clamp_only_positive(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  // Testing with negative infinity.
  %0 = "ttir.clamp_scalar"(%arg0) <{max = 1.000000e+00 : f32, min = 0xFF800000 : f32}> : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %0 : tensor<64x128xbf16>
}

func.func @clamp_only_negative(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  // Testing with positive infinity.
  %0 = "ttir.clamp_scalar"(%arg0) <{max = 0x7F800000 : f32, min = -1.000000e+00 : f32}> : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %0 : tensor<64x128xbf16>
}

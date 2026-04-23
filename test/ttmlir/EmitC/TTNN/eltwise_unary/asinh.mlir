// TODO(dmilinkovicTT): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @asinh_f32(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %1 = "ttir.asinh"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @asinh_bf16(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %1 = "ttir.asinh"(%arg0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}

func.func @asinh_rank3(%arg0: tensor<2x64x128xf32>) -> tensor<2x64x128xf32> {
  %1 = "ttir.asinh"(%arg0) : (tensor<2x64x128xf32>) -> tensor<2x64x128xf32>
  return %1 : tensor<2x64x128xf32>
}

func.func @asinh_rank4(%arg0: tensor<2x4x64x128xf32>) -> tensor<2x4x64x128xf32> {
  %1 = "ttir.asinh"(%arg0) : (tensor<2x4x64x128xf32>) -> tensor<2x4x64x128xf32>
  return %1 : tensor<2x4x64x128xf32>
}

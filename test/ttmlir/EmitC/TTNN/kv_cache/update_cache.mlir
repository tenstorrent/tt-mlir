// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_fb.mlir
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @update_cache(%arg0: tensor<1x8x16x128xbf16>, %arg1: tensor<1x8x1x128xbf16>, %arg2: tensor<1xi32>) -> tensor<1x8x16x128xbf16> {
  %0 = "ttir.update_cache"(%arg0, %arg1, %arg2) <{batch_offset = 0 : i32}> : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi32>) -> tensor<1x8x16x128xbf16>
  return %0 : tensor<1x8x16x128xbf16>
}

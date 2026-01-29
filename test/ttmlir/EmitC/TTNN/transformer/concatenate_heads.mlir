// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @concatenate_heads(%arg0: tensor<1x24x32x128xbf16>) -> tensor<1x32x3072xbf16> {
  %1 = "ttir.concatenate_heads"(%arg0) : (tensor<1x24x32x128xbf16>) -> tensor<1x32x3072xbf16>
  return %1 : tensor<1x32x3072xbf16>
}
func.func @concatenate_heads_batch_2(%arg0: tensor<2x24x32x128xbf16>) -> tensor<2x32x3072xbf16> {
  %1 = "ttir.concatenate_heads"(%arg0) : (tensor<2x24x32x128xbf16>) -> tensor<2x32x3072xbf16>
  return %1 : tensor<2x32x3072xbf16>
}

// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t_fb.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @split_qkv_and_split_heads(%arg0: tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) {
  %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>)
  return %3, %4, %5 : tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>
}

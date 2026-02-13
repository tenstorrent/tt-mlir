// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_fb.mlir
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func public @argmax_2d(%arg0: tensor<64x64xf32>) -> tensor<64xi32> {
  // CHECK-LABEL: func.func public @argmax_2d(
  // CHECK: "ttnn.argmax"
  // CHECK-SAME: {dim = 1 : i32, use_multicore = false}>
  // CHECK-SAME: tensor<64x64xf32
  // CHECK-SAME: tensor<64xi32
  // CHECK-SAME: -> tensor<64xi32
  %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<64x64xf32>) -> tensor<64xi32>
  return %0 : tensor<64xi32>
}

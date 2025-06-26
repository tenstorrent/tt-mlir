// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func public @argmax_2d(%arg0: tensor<64x64xf32>) -> tensor<64xi32> {
  // CHECK-LABEL: func.func public @argmax_2d(
  %0 = ttir.empty() : tensor<64xi32>
  // CHECK: "ttnn.argmax"
  // CHECK-SAME: {dim = 1 : i32, use_multicore = false}>
  // CHECK-SAME: tensor<64x64xf32
  // CHECK-SAME: tensor<64xi32
  // CHECK-SAME: -> tensor<64xi32
  %1 = "ttir.argmax"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<64x64xf32>, tensor<64xi32>) -> tensor<64xi32>
  return %1 : tensor<64xi32>
}

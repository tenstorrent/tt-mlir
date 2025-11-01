// UNSUPPORTED: true
// Reason: Hangs when running on silicon - likely due to https://github.com/tenstorrent/tt-metal/pull/31511
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func public @batch_norm_training(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) {
    %0 = ttir.empty() : tensor<2x2x2x2xf32>
    %1 = ttir.empty() : tensor<2xf32>
    %2 = ttir.empty() : tensor<2xf32>
    %result, %batch_mean, %batch_variance = "ttir.batch_norm_training"(%arg0, %arg1, %arg2, %arg3, %arg4, %0, %1, %2) <{dimension = 1 : i32, epsilon = 0.000000e+00 : f32, momentum = 1.000000e-01 : f32}> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
    return %result, %batch_mean, %batch_variance : tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>
  }
}

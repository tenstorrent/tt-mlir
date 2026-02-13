// REQUIRES: stablehlo
// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_fb.mlir
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

// default
func.func @scatter(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
  }) : (tensor<1x3x320x320xf32>, tensor<1x1xi32>, tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32>
  return %0 : tensor<1x3x320x320xf32>
}

// gpt-oss - multi-dimensional scatter with scatter operation broken into multiple scatter operations each handling index_shape[dim] < 256
func.func @scatter_1(%arg0: tensor<71x32xbf16>, %arg1: tensor<71x4x2xi64>, %arg2: tensor<71x4xbf16>) -> tensor<71x32xbf16> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
    stablehlo.return %arg4 : tensor<bf16>
  }) : (tensor<71x32xbf16>, tensor<71x4x2xi64>, tensor<71x4xbf16>) -> tensor<71x32xbf16>
  return %0 : tensor<71x32xbf16>
}

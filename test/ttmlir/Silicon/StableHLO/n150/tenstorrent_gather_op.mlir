// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// End-to-end check that a `stablehlo.composite "tenstorrent.gather_dim"` lowers
// all the way to `ttnn.gather` and produces a valid flatbuffer.

module attributes {} {
  // CHECK-LABEL: func.func @gather_dim_2d
  func.func @gather_dim_2d(%arg0: tensor<8x4xf32>, %arg1: tensor<3x4xui32>) -> tensor<3x4xf32> {
    // CHECK: "ttnn.gather"
    %0 = stablehlo.composite "tenstorrent.gather_dim" %arg0, %arg1 {
      composite_attributes = {dim = 0 : i64},
      decomposition = @tenstorrent.gather_dim.impl_2d
    } : (tensor<8x4xf32>, tensor<3x4xui32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
  func.func private @tenstorrent.gather_dim.impl_2d(%arg0: tensor<8x4xf32>, %arg1: tensor<3x4xui32>) -> tensor<3x4xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @gather_dim_1d
  func.func @gather_dim_1d(%arg0: tensor<8xf32>, %arg1: tensor<3xui32>) -> tensor<3xf32> {
    // CHECK: "ttnn.gather"
    %0 = stablehlo.composite "tenstorrent.gather_dim" %arg0, %arg1 {
      composite_attributes = {dim = 0 : i64},
      decomposition = @tenstorrent.gather_dim.impl_1d
    } : (tensor<8xf32>, tensor<3xui32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
  func.func private @tenstorrent.gather_dim.impl_1d(%arg0: tensor<8xf32>, %arg1: tensor<3xui32>) -> tensor<3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @gather_composite_tests attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {

  // Test: Gather along dim 0 with i32 index — index gets typecast to ui32.
  // CHECK-LABEL: func.func @gather_dim0_i32_index
  func.func @gather_dim0_i32_index(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    // CHECK: "ttir.typecast"(%arg1)
    // CHECK-SAME: (tensor<2x3xi32>) -> tensor<2x3xui32>
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = 0 : i32
    // CHECK-SAME: sparse_grad = false
    // CHECK-SAME: (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_dim0_i32} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  func.func private @tenstorrent.gather.impl_dim0_i32(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Test: Gather along dim 1 with i64 index — index gets typecast to ui32.
  // CHECK-LABEL: func.func @gather_dim1_i64_index
  func.func @gather_dim1_i64_index(%arg0: tensor<3x5xbf16>, %arg1: tensor<3x2xi64>) -> tensor<3x2xbf16> {
    // CHECK: "ttir.typecast"(%arg1)
    // CHECK-SAME: (tensor<3x2xi64>) -> tensor<3x2xui32>
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = 1 : i32
    // CHECK-SAME: sparse_grad = false
    // CHECK-SAME: (tensor<3x5xbf16>, tensor<3x2xui32>) -> tensor<3x2xbf16>
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 1 : i64}, decomposition = @tenstorrent.gather.impl_dim1_i64} : (tensor<3x5xbf16>, tensor<3x2xi64>) -> tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
  func.func private @tenstorrent.gather.impl_dim1_i64(%arg0: tensor<3x5xbf16>, %arg1: tensor<3x2xi64>) -> tensor<3x2xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }

  // Test: Gather with negative dim.
  // CHECK-LABEL: func.func @gather_negative_dim
  func.func @gather_negative_dim(%arg0: tensor<4x6xf32>, %arg1: tensor<4x3xi32>) -> tensor<4x3xf32> {
    // CHECK: "ttir.typecast"(%arg1)
    // CHECK-SAME: (tensor<4x3xi32>) -> tensor<4x3xui32>
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = -1 : i32
    // CHECK-SAME: (tensor<4x6xf32>, tensor<4x3xui32>) -> tensor<4x3xf32>
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = -1 : i64}, decomposition = @tenstorrent.gather.impl_neg_dim} : (tensor<4x6xf32>, tensor<4x3xi32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
  func.func private @tenstorrent.gather.impl_neg_dim(%arg0: tensor<4x6xf32>, %arg1: tensor<4x3xi32>) -> tensor<4x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }

  // Test: Gather with 3D tensors along dim 2.
  // CHECK-LABEL: func.func @gather_3d
  func.func @gather_3d(%arg0: tensor<2x4x6xf32>, %arg1: tensor<2x4x3xi32>) -> tensor<2x4x3xf32> {
    // CHECK: "ttir.typecast"(%arg1)
    // CHECK-SAME: (tensor<2x4x3xi32>) -> tensor<2x4x3xui32>
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = 2 : i32
    // CHECK-SAME: (tensor<2x4x6xf32>, tensor<2x4x3xui32>) -> tensor<2x4x3xf32>
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 2 : i64}, decomposition = @tenstorrent.gather.impl_3d} : (tensor<2x4x6xf32>, tensor<2x4x3xi32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
  func.func private @tenstorrent.gather.impl_3d(%arg0: tensor<2x4x6xf32>, %arg1: tensor<2x4x3xi32>) -> tensor<2x4x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }

  // Test: Gather with ui32 index — no typecast should be inserted.
  // CHECK-LABEL: func.func @gather_ui32_index
  func.func @gather_ui32_index(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    // CHECK-NOT: "ttir.typecast"
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = 0 : i32
    // CHECK-SAME: (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_ui32} : (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  func.func private @tenstorrent.gather.impl_ui32(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Test: Gather with ui16 index — no typecast should be inserted.
  // CHECK-LABEL: func.func @gather_ui16_index
  func.func @gather_ui16_index(%arg0: tensor<5x3xbf16>, %arg1: tensor<2x3xui16>) -> tensor<2x3xbf16> {
    // CHECK-NOT: "ttir.typecast"
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = 0 : i32
    // CHECK-SAME: (tensor<5x3xbf16>, tensor<2x3xui16>) -> tensor<2x3xbf16>
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_ui16} : (tensor<5x3xbf16>, tensor<2x3xui16>) -> tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
  func.func private @tenstorrent.gather.impl_ui16(%arg0: tensor<5x3xbf16>, %arg1: tensor<2x3xui16>) -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }

  // Test: Gather with sparse_grad attribute set to true.
  // CHECK-LABEL: func.func @gather_sparse_grad
  func.func @gather_sparse_grad(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    // CHECK: "ttir.typecast"(%arg1)
    // CHECK-SAME: (tensor<2x3xi32>) -> tensor<2x3xui32>
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = 0 : i32
    // CHECK-SAME: sparse_grad = true
    // CHECK-SAME: (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64, sparse_grad = true}, decomposition = @tenstorrent.gather.impl_sparse} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  func.func private @tenstorrent.gather.impl_sparse(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Test: Gather with no composite_attributes — defaults to dim=0, sparse_grad=false.
  // CHECK-LABEL: func.func @gather_default_attrs
  func.func @gather_default_attrs(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    // CHECK: "ttir.typecast"(%arg1)
    // CHECK-SAME: (tensor<2x3xi32>) -> tensor<2x3xui32>
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = 0 : i32
    // CHECK-SAME: sparse_grad = false
    // CHECK-SAME: (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {decomposition = @tenstorrent.gather.impl_default} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  func.func private @tenstorrent.gather.impl_default(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

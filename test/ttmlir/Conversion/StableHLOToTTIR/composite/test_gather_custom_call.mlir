// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -split-input-file -o %t %s
// RUN: FileCheck %s --input-file=%t

// Exercises CustomCallGatherConversionPattern: stablehlo.custom_call
// @tenstorrent.gather_dim with tt.has_custom_sharding -> ttir.gather. This is
// the post-flatten path that runs when FlattenOrConvertCompositesPass has
// already turned the composite into a custom_call so Shardy can propagate
// shardings through it.

// Test: dim 0 with i32 index -- index is typecast to ui32.
module @gather_dim_dim0_i32 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    // CHECK: "ttir.typecast"(%arg1)
    // CHECK-SAME: (tensor<2x3xi32>) -> tensor<2x3xui32>
    // CHECK: "ttir.gather"
    // CHECK-SAME: dim = 0 : i32
    // CHECK-SAME: (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    %0 = stablehlo.custom_call @tenstorrent.gather_dim(%arg0, %arg1) {
      tt.composite_attributes = {dim = 0 : i64},
      tt.has_custom_sharding
    } : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// Test: dim 1 with i64 index -- index is typecast to ui32.
module @gather_dim_dim1_i64 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<3x5xbf16>, %arg1: tensor<3x2xi64>) -> tensor<3x2xbf16> {
    // CHECK: "ttir.typecast"(%arg1)
    // CHECK-SAME: (tensor<3x2xi64>) -> tensor<3x2xui32>
    // CHECK: "ttir.gather"
    // CHECK-SAME: dim = 1 : i32
    // CHECK-SAME: (tensor<3x5xbf16>, tensor<3x2xui32>) -> tensor<3x2xbf16>
    %0 = stablehlo.custom_call @tenstorrent.gather_dim(%arg0, %arg1) {
      tt.composite_attributes = {dim = 1 : i64},
      tt.has_custom_sharding
    } : (tensor<3x5xbf16>, tensor<3x2xi64>) -> tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}

// -----

// Test: negative dim with ui32 index -- no typecast inserted.
module @gather_dim_negative attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<4x6xf32>, %arg1: tensor<4x3xui32>) -> tensor<4x3xf32> {
    // CHECK-NOT: "ttir.typecast"
    // CHECK: "ttir.gather"
    // CHECK-SAME: dim = -1 : i32
    // CHECK-SAME: (tensor<4x6xf32>, tensor<4x3xui32>) -> tensor<4x3xf32>
    %0 = stablehlo.custom_call @tenstorrent.gather_dim(%arg0, %arg1) {
      tt.composite_attributes = {dim = -1 : i64},
      tt.has_custom_sharding
    } : (tensor<4x6xf32>, tensor<4x3xui32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

// -----

// Test: 3D tensor with ui16 index -- no typecast inserted.
module @gather_dim_3d_ui16 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<2x4x6xf32>, %arg1: tensor<2x4x3xui16>) -> tensor<2x4x3xf32> {
    // CHECK-NOT: "ttir.typecast"
    // CHECK: "ttir.gather"
    // CHECK-SAME: dim = 2 : i32
    // CHECK-SAME: (tensor<2x4x6xf32>, tensor<2x4x3xui16>) -> tensor<2x4x3xf32>
    %0 = stablehlo.custom_call @tenstorrent.gather_dim(%arg0, %arg1) {
      tt.composite_attributes = {dim = 2 : i64},
      tt.has_custom_sharding
    } : (tensor<2x4x6xf32>, tensor<2x4x3xui16>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}

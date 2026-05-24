// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// =====================================================================
// Gather with sharding on a non-gather dim: input and index dim 0 are linked
// via a kPassThrough factor, so the sharding propagates to the output with
// no collectives needed.
// =====================================================================
module @Gather_ShardNonGatherDim attributes {mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22batch\22=2]>}"}} {
  func.func @main(
    %arg0: tensor<4x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22batch\22}, {}]>"}},
    %arg1: tensor<4x5xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22batch\22}, {}]>"}}
  ) -> tensor<4x5xf32> {
    // CHECK-LABEL: @main
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tenstorrent.gather_dim
    // CHECK-SAME: (tensor<2x8xf32>, tensor<2x5xi64>) -> tensor<2x5xf32>
    %0 = stablehlo.custom_call @tenstorrent.gather_dim(%arg0, %arg1) {
      api_version = 0 : i32,
      tt.composite_attributes = {dim = 1 : i64},
      tt.has_custom_sharding
    } : (tensor<4x8xf32>, tensor<4x5xi64>) -> tensor<4x5xf32>
    return %0 : tensor<4x5xf32>
  }
}

// -----

// =====================================================================
// Gather with sharding on the input's gather dim: the input gather dim is
// kNeedReplication, so Shardy inserts an all_gather to replicate it before
// the gather runs with full input data on each device.
// =====================================================================
module @Gather_ShardInputGatherDim attributes {mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22batch\22=2]>}"}} {
  func.func @main(
    %arg0: tensor<8x4xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22batch\22}, {}]>"}},
    %arg1: tensor<3x4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}}
  ) -> tensor<3x4xf32> {
    // CHECK-LABEL: @main
    // CHECK: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tenstorrent.gather_dim
    // CHECK-SAME: (tensor<8x4xf32>, tensor<3x4xi64>) -> tensor<3x4xf32>
    %0 = stablehlo.custom_call @tenstorrent.gather_dim(%arg0, %arg1) {
      api_version = 0 : i32,
      tt.composite_attributes = {dim = 0 : i64},
      tt.has_custom_sharding
    } : (tensor<8x4xf32>, tensor<3x4xi64>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}

// -----

// =====================================================================
// Gather with sharding on the index's gather dim: the index gather dim is
// kPassThrough and is independent of the input gather dim. The sharding
// propagates to the output's gather dim with no collectives needed.
// =====================================================================
module @Gather_ShardIndexGatherDim attributes {mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22batch\22=2]>}"}} {
  func.func @main(
    %arg0: tensor<8x4xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}},
    %arg1: tensor<8x4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22batch\22}]>"}}
  ) -> tensor<8x4xf32> {
    // CHECK-LABEL: @main
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tenstorrent.gather_dim
    // CHECK-SAME: (tensor<8x4xf32>, tensor<8x2xi64>) -> tensor<8x2xf32>
    %0 = stablehlo.custom_call @tenstorrent.gather_dim(%arg0, %arg1) {
      api_version = 0 : i32,
      tt.composite_attributes = {dim = 1 : i64},
      tt.has_custom_sharding
    } : (tensor<8x4xf32>, tensor<8x4xi64>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}

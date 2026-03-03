// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// CHECK-LABEL: module @SyncTensorsGraph.5
module @SyncTensorsGraph.5 attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  // CHECK: sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  func.func @main(%arg0: tensor<32x128xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}"}, %arg1: tensor<32x32xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}"}) -> tensor<32x128xf32> {
    // CHECK: sdy.manual_computation(%{{.*}}, %{{.*}}) in_shardings=[<@mesh, [{}, {"_axis_0"}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {"_axis_0"}]>] manual_axes={"_axis_0_updated", "_axis_0"}
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }
}

// -----

// CHECK-LABEL: module @ShardingConstraint
module @ShardingConstraint attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  // CHECK: sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  func.func @main(%arg0: tensor<1x8x16x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}"}) -> tensor<1x8x16x128xbf16> {
    // CHECK: sdy.manual_computation(%{{.*}}) in_shardings=[<@mesh, [{}, {"_axis_0"}, {}, {}]>] out_shardings=[<@mesh, [{}, {"_axis_0"}, {}, {}]>] manual_axes={"_axis_0_updated", "_axis_0"}
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {\22_axis_0\22}, {}, {}]>]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}"} : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xbf16>
    return %0 : tensor<1x8x16x128xbf16>
  }
}

// -----

// CHECK-LABEL: module @ScalarShardingConstraint
// Test scalar tensor (rank 0) with empty dimension shardings
module @ScalarShardingConstraint attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  // CHECK: sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  func.func @main(%arg0: tensor<i1> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, []>"}, mhlo.sharding = "{replicated}"}) -> tensor<i1> {
    // CHECK: sdy.manual_computation(%{{.*}}) in_shardings=[<@mesh, []>] out_shardings=[<@mesh, []>] manual_axes={"_axis_0_updated", "_axis_0"}
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, []>]>"}, mhlo.sharding = "{replicated}"} : (tensor<i1>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}


// -----

// CHECK-LABEL: module @TTShardingConstraint
module @TTShardingConstraint attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2, \22_axis_1\22=4]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x128xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}", ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<131072x8192xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_1\22}]>"}, mhlo.sharding = "{devices=[1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}", ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<2x128x8192xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<2x128xi64>) -> tensor<256xi64>
    %1 = stablehlo.convert %0 : (tensor<256xi64>) -> tensor<256xui32>
    %2 = "stablehlo.gather"(%arg1, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8192>}> : (tensor<131072x8192xf32>, tensor<256xui32>) -> tensor<256x8192xf32>
    %3 = stablehlo.reshape %2 : (tensor<256x8192xf32>) -> tensor<2x128x8192xf32>
    // CHECK: stablehlo.all_gather
    // CHECK-SAME: (tensor<1x128x2048xf32>
    // CHECK-SAME: -> tensor<1x128x8192xf32>
    %4 = stablehlo.custom_call @tt.sharding_constraint(%3) {api_version = 0 : i32, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22mesh_idx_0\22}, {}, {}]>]>"}} : (tensor<2x128x8192xf32>) -> tensor<2x128x8192xf32>
    return %4 : tensor<2x128x8192xf32>
  }
}

// -----

// CHECK-LABEL: module @UserPriority
// Test priority-based sharding propagation:
// - arg1 (input_ids): batch dim sharded with priority 0 (propagates first)
// - arg2 (embedding_weight): hidden dim sharded with priority 1 (propagates second)
// Expected: batch sharding wins, requiring all_gather -> gather -> all_to_all
module @UserPriority attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2, \22_axis_1\22=4]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(
    %arg0: tensor<512xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<parameter>},
    %arg1: tensor<2x32xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}p0, {}]>"}, mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}", ttcore.argument_type = #ttcore.argument_type<input>},
    %arg2: tensor<1000x512xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}p1]>"}, mhlo.sharding = "{devices=[1,2,4]<=[8] last_tile_dim_replicate}", ttcore.argument_type = #ttcore.argument_type<parameter>}
  ) -> tensor<2x32x512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.001953125> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<2x32x1xf32>
    %1 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x32xf32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x32x512xf32>
    %3 = stablehlo.reshape %arg2 : (tensor<1000x512xf32>) -> tensor<1x1000x512xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x1000x512xf32>) -> tensor<1000x512xf32>
    %5 = stablehlo.reshape %arg1 : (tensor<2x32xi64>) -> tensor<1x2x32xi64>
    %6 = stablehlo.reshape %5 : (tensor<1x2x32xi64>) -> tensor<64xi64>
    %7 = stablehlo.convert %6 : (tensor<64xi64>) -> tensor<64xui32>
    %8 = "stablehlo.gather"(%4, %7) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 512>}> : (tensor<1000x512xf32>, tensor<64xui32>) -> tensor<64x512xf32>
    // CHECK: stablehlo.all_gather
    // CHECK-SAME: -> tensor<64x
    // CHECK: stablehlo.gather
    // CHECK-SAME: tensor<1000x256x
    // CHECK-SAME: tensor<64x
    // CHECK-SAME: -> tensor<64x256x
    // CHECK: stablehlo.all_to_all
    // CHECK-SAME: (tensor<64x256x
    // CHECK-SAME: -> tensor<32x512x
    %9 = stablehlo.reshape %8 : (tensor<64x512xf32>) -> tensor<2x32x512xf32>
    %10 = stablehlo.power %9, %2 : tensor<2x32x512xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x32x512xf32>, tensor<f32>) -> tensor<2x32xf32>
    %12 = stablehlo.multiply %11, %1 : tensor<2x32xf32>
    %13 = stablehlo.reshape %12 : (tensor<2x32xf32>) -> tensor<2x32x1xf32>
    %14 = stablehlo.add %13, %0 : tensor<2x32x1xf32>
    %15 = stablehlo.rsqrt %14 : tensor<2x32x1xf32>
    %16 = stablehlo.reshape %15 : (tensor<2x32x1xf32>) -> tensor<2x32xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<2x32xf32>) -> tensor<2x32x512xf32>
    %18 = stablehlo.multiply %9, %17 : tensor<2x32x512xf32>
    %19 = stablehlo.reshape %arg0 : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %20 = stablehlo.reshape %19 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [2] : (tensor<512xf32>) -> tensor<2x32x512xf32>
    %22 = stablehlo.multiply %18, %21 : tensor<2x32x512xf32>
    return %22 : tensor<2x32x512xf32>
  }
}

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

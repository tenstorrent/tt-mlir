// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// tt.topk_large_indices layout:
//   input [rows, N] -> indices [rows, k]   (top-k over the last dim N).

// Data parallelism over rows. The input/output shard on the leading (row) dim;
// the row-length dim N stays replicated (the fused op needs the whole row), so
// the op runs entirely on local row-shards with no collective.
// CHECK-LABEL: module @TopKLargeIndices_Sharding_Rows
module @TopKLargeIndices_Sharding_Rows attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<4x256xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"}) -> tensor<4x64xui32> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tt.topk_large_indices
    // CHECK-SAME: tensor<2x256xbf16>
    // CHECK-SAME: -> tensor<2x64xui32>
    %0 = stablehlo.custom_call @tt.topk_large_indices(%arg0) {api_version = 0 : i32, mhlo.frontend_attributes = {k = "64"}} : (tensor<4x256xbf16>) -> tensor<4x64xui32>
    return %0 : tensor<4x64xui32>
  }
}

// -----

// Sharding the row-length dim N. Because N is kNeedReplication (the fused op
// needs the whole row on one device), the partitioner inserts an all_gather to
// replicate N before the op; the op then runs on the full row length.
// CHECK-LABEL: module @TopKLargeIndices_Sharding_RowLen
module @TopKLargeIndices_Sharding_RowLen attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<4x256xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"}) -> tensor<4x64xui32> {
    // CHECK: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.topk_large_indices
    // CHECK-SAME: tensor<4x256xbf16>
    // CHECK-SAME: -> tensor<4x64xui32>
    %0 = stablehlo.custom_call @tt.topk_large_indices(%arg0) {api_version = 0 : i32, mhlo.frontend_attributes = {k = "64"}} : (tensor<4x256xbf16>) -> tensor<4x64xui32>
    return %0 : tensor<4x64xui32>
  }
}

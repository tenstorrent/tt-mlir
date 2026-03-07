// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// =====================================================================
// sparse_matmul (is_input_b_sparse): K is kReduction → inserts all_reduce
// =====================================================================
// Input A is sharded on K (dim 3, contracting dim) via _axis_0. Weight B also
// has K at dim 2. The K factor uses kReduction, so after matmul Shardy inserts
// all_reduce to sum partial results across devices.
module @SparseMatmul_KReduction attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x4x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input_a"}, %arg1: tensor<1x4x64x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "input_b"}, %arg2: tensor<2x4x1x4xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "sparsity"}) -> tensor<2x4x1x4x32x128xbf16> {
    // K is the contracting dimension → kReduction → all_reduce after sparse_matmul
    // CHECK: stablehlo.custom_call @tt.sparse_matmul
    // CHECK: stablehlo.all_reduce
    %0 = stablehlo.custom_call @tt.sparse_matmul(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {is_input_a_sparse = "False", is_input_b_sparse = "True"}} : (tensor<2x4x32x64xbf16>, tensor<1x4x64x128xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x128xbf16>
    return %0 : tensor<2x4x1x4x32x128xbf16>
  }
}

// -----

// =====================================================================
// sparse_matmul (is_input_b_sparse): EP on E is kPassThrough → no CCL
// =====================================================================
// Input B is sharded on E (dim 1) via _axis_0 (expert parallelism).
// The E factor uses kPassThrough, so EP propagates through the op.
module @SparseMatmul_EPPassthrough attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x4x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input_a"}, %arg1: tensor<1x4x64x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "input_b"}, %arg2: tensor<2x4x1x4xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "sparsity"}) -> tensor<2x4x1x4x32x128xbf16> {
    // E is kPassThrough: no CCL needed for EP.
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tt.sparse_matmul
    %0 = stablehlo.custom_call @tt.sparse_matmul(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {is_input_a_sparse = "False", is_input_b_sparse = "True"}} : (tensor<2x4x32x64xbf16>, tensor<1x4x64x128xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x128xbf16>
    return %0 : tensor<2x4x1x4x32x128xbf16>
  }
}

// -----

// =====================================================================
// sparse_matmul (is_input_b_sparse): compound EP on E across 2D mesh
// =====================================================================
// On a (2,4) mesh, E=8 is compound-sharded across both axes:
//   Input B [1,8,K,N]: E (dim 1) sharded as (_axis_0, _axis_1) -> 8/(2*4)=1 local
//   Sparsity [A,B,1,8]: E (dim 3) sharded as (_axis_0, _axis_1)
// E factor is kPassThrough, so compound sharding propagates to output E (dim 3).
// No all_gather or all_reduce needed -- each device computes its local expert slice.
module @SparseMatmul_CompoundEP attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2, \22_axis_1\22=4]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x4x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input_a"}, %arg1: tensor<1x8x64x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22, \22_axis_1\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,8,1,1]<=[8]}", ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "input_b"}, %arg2: tensor<2x4x1x8xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22, \22_axis_1\22}]>"}, mhlo.sharding = "{devices=[1,1,1,8]<=[8]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "sparsity"}) -> tensor<2x4x1x8x32x128xbf16> {
    // Compound EP: E sharded across (_axis_0, _axis_1). kPassThrough propagates
    // the compound sharding to the output without any CCL insertion.
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tt.sparse_matmul
    %0 = stablehlo.custom_call @tt.sparse_matmul(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {is_input_a_sparse = "False", is_input_b_sparse = "True"}} : (tensor<2x4x32x64xbf16>, tensor<1x8x64x128xbf16>, tensor<2x4x1x8xbf16>) -> tensor<2x4x1x8x32x128xbf16>
    return %0 : tensor<2x4x1x8x32x128xbf16>
  }
}

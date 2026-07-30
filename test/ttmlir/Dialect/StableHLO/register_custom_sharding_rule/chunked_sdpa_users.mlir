// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Chunked SDPA with the query sharded on the DP axis. The users factor must
// carry that sharding to the result, so the partitioned result keeps the
// per-shard user count and no all_gather is needed.
module @chunked_sdpa_users_dp attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["batch"=2, "model"=2]>
  func.func @main(
    %arg0: tensor<8x4x64x16xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"},
    %arg1: tensor<32x2x32x16xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"},
    %arg2: tensor<32x2x32x16xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"},
    %arg3: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "page_table"},
    %arg4: tensor<1xi32> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "chunk_start_idx"}
  ) -> tensor<8x4x64x16xbf16> {
    %0 = stablehlo.custom_call @tt.chunked_scaled_dot_product_attention(%arg0, %arg1, %arg2, %arg3, %arg4) {mhlo.frontend_attributes = {scale = "0.25"}} : (tensor<8x4x64x16xbf16>, tensor<32x2x32x16xbf16>, tensor<32x2x32x16xbf16>, tensor<8x4xi32>, tensor<1xi32>) -> tensor<8x4x64x16xbf16>
    return %0 : tensor<8x4x64x16xbf16>
  }
}

// The result must keep "batch" on dim 0, i.e. both operand and result are
// partitioned to 4 users per shard rather than the result widening back to 8.
// CHECK: tensor<4x2x64x16xbf16>
// CHECK-NOT: stablehlo.all_gather

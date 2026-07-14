// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --flatten-or-convert-composites -o %t %s
// RUN: FileCheck %s --input-file=%t

// Regression for the #8601 sharding regression (tt-xla DeepSeek-V3 TP galaxy
// `new_volume == old_volume` reshape crash).
//
// getTopKShardingRule() only builds a valid rule for the rank-2 [batch, N]
// form. DeepSeek-V3's MoE gate emits a rank-3 [tokens, groups, experts] topk
// that is sharded on the batch (tokens) dim, not the reduction (experts) dim.
// Converting it to a custom_call leaves Shardy with an empty rule, so the batch
// sharding is not carried across the op and the global batch dim corrupts the
// local shard. FlattenOrConvertCompositesPass must instead FLATTEN a rank-3
// topk (no stablehlo.custom_call, no stablehlo.composite remaining).
module @jit_topk_rank3 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["batch"=4, "vocab"=1]>
  // CHECK-LABEL: func.func @main
  // CHECK-NOT: stablehlo.custom_call @tenstorrent.topk
  // CHECK-NOT: stablehlo.composite "tenstorrent.topk"
  // CHECK: stablehlo.slice
  func.func @main(
    %arg0: tensor<256x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>, ttcore.shard_status = #ttcore.shard_status<presharded>}
  ) -> (tensor<256x8x2xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>},
        tensor<256x8x2xi64> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0:2 = stablehlo.composite "tenstorrent.topk" %arg0 {
      composite_attributes = {dim = -1 : i64, k = 2 : i64, largest = true, sorted = true},
      decomposition = @tenstorrent.topk.impl_rank3
    } : (tensor<256x8x32xf32>) -> (tensor<256x8x2xf32>, tensor<256x8x2xi64>)
    return %0#0, %0#1 : tensor<256x8x2xf32>, tensor<256x8x2xi64>
  }
  func.func private @tenstorrent.topk.impl_rank3(%arg0: tensor<256x8x32xf32>) -> (tensor<256x8x2xf32>, tensor<256x8x2xi64>) {
    %values = stablehlo.slice %arg0 [0:256, 0:8, 0:2] : (tensor<256x8x32xf32>) -> tensor<256x8x2xf32>
    %indices = stablehlo.constant dense<0> : tensor<256x8x2xi64>
    return %values, %indices : tensor<256x8x2xf32>, tensor<256x8x2xi64>
  }
}

// -----

// A rank-2 [batch, N] topk is the form getTopKShardingRule() supports, so it
// must still be converted to a stablehlo.custom_call so Shardy can propagate
// its custom sharding rule (the intended #8601 path). This is the vLLM
// vocab-sharded sampling case, where the reduction dim N is the sharded one.
module @jit_topk_rank2 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["batch"=1, "vocab"=4]>
  // CHECK-LABEL: func.func @main
  // CHECK: stablehlo.custom_call @tenstorrent.topk
  // CHECK-SAME: tt.has_custom_sharding
  // CHECK-SAME: (tensor<8x256xf32>) -> (tensor<8x2xf32>, tensor<8x2xi64>)
  // CHECK-NOT: stablehlo.composite "tenstorrent.topk"
  func.func @main(
    %arg0: tensor<8x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"vocab"}]>, ttcore.shard_status = #ttcore.shard_status<presharded>}
  ) -> (tensor<8x2xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>},
        tensor<8x2xi64> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0:2 = stablehlo.composite "tenstorrent.topk" %arg0 {
      composite_attributes = {dim = -1 : i64, k = 2 : i64, largest = true, sorted = true},
      decomposition = @tenstorrent.topk.impl_rank2
    } : (tensor<8x256xf32>) -> (tensor<8x2xf32>, tensor<8x2xi64>)
    return %0#0, %0#1 : tensor<8x2xf32>, tensor<8x2xi64>
  }
  func.func private @tenstorrent.topk.impl_rank2(%arg0: tensor<8x256xf32>) -> (tensor<8x2xf32>, tensor<8x2xi64>) {
    %values = stablehlo.slice %arg0 [0:8, 0:2] : (tensor<8x256xf32>) -> tensor<8x2xf32>
    %indices = stablehlo.constant dense<0> : tensor<8x2xi64>
    return %values, %indices : tensor<8x2xf32>, tensor<8x2xi64>
  }
}

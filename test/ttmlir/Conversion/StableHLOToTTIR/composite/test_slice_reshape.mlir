// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// The Tenstorrent slice_reshape composite legalizes to ttir.slice_reshape,
// carrying its `begins`, `ends`, `step`, and `shape` composite_attributes
// through onto the TTIR op.

module @jit__slice_reshape attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <["default_updated"=1, "default"=1]>

  // Quetzal pattern: getitem(slice along last dim) -> reshape (e.g. fused
  // QKV linear feeding multi-head reshape). The slice keeps rank but
  // narrows the last dim, then the reshape fans the embed-dim into
  // (heads, head_dim).
  // CHECK-LABEL: func.func public @test_slice_reshape
  // CHECK: "ttir.slice_reshape"
  // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32]
  // CHECK-SAME: ends = [1 : i32, 1 : i32, 2048 : i32]
  // CHECK-SAME: shape = [1 : i32, 32 : i32, 64 : i32]
  // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32]
  func.func public @test_slice_reshape(%arg0: tensor<1x1x6144xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<1x32x64xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = stablehlo.composite "tenstorrent.slice_reshape" %arg0 {
      composite_attributes = {
        begins = [0 : i32, 0 : i32, 0 : i32],
        ends = [1 : i32, 1 : i32, 2048 : i32],
        step = [1 : i32, 1 : i32, 1 : i32],
        shape = [1 : i32, 32 : i32, 64 : i32]
      },
      decomposition = @tenstorrent.slice_reshape.impl
    } : (tensor<1x1x6144xf32>) -> tensor<1x32x64xf32>
    return %0 : tensor<1x32x64xf32>
  }

  // The decomposition body is a stand-in; the legalize pass never inspects
  // it but stablehlo.composite requires the symbol to exist.
  func.func private @tenstorrent.slice_reshape.impl(%arg0: tensor<1x1x6144xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<1x32x64xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x32x64xf32>
    return %cst : tensor<1x32x64xf32>
  }
}

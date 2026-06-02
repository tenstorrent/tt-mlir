// RUN: ttmlir-opt --ttcore-register-device --const-eval-hoist-transform %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// Verifies the L1 const-eval infra:
//   (1) An L1-resident op tagged with `ttnn.const_eval_allowed` is hoisted
//       into a const-eval function.
//   (2) The pass records the per-core L1 footprint of const-eval outputs in
//       the module attribute `ttnn.l1_const_eval_usage`.
//
// Sharded layout footprint per core = getShardSizeInBytes() =
//   shard_shape (1x1 tiles) * tile (32x32 bf16) = 32*32*2 bytes = 2048 bytes;
//   aligned with cushion = 2048 + 1024 = 3072 bytes.

#l1 = #ttnn.buffer_type<l1>

#l1_sharded = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0, 0), (0, 0)>]>>

// CHECK: module attributes {{.*}}ttnn.l1_const_eval_usage = 3072 : ui64
module {
  // CHECK-LABEL: func.func private @tagged_l1_const_eval_0
  // CHECK: "ttnn.add"
  // CHECK: return

  // CHECK-LABEL: func.func @tagged_l1(
  func.func @tagged_l1(
      %arg0: tensor<32x32xbf16, #l1_sharded> {ttcore.argument_type = #ttcore.argument_type<input>},
      %arg1: tensor<32x32xbf16, #l1_sharded> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %arg2: tensor<32x32xbf16, #l1_sharded> {ttcore.argument_type = #ttcore.argument_type<parameter>}
  ) -> tensor<32x32xbf16, #l1_sharded> {
    // CHECK: ttcore.load_cached(@tagged_l1_const_eval_0, [%arg1, %arg2])
    %tagged = "ttnn.add"(%arg1, %arg2) {ttnn.const_eval_allowed} : (tensor<32x32xbf16, #l1_sharded>, tensor<32x32xbf16, #l1_sharded>) -> tensor<32x32xbf16, #l1_sharded>
    // CHECK: "ttnn.add"(%arg0
    %result = "ttnn.add"(%arg0, %tagged) : (tensor<32x32xbf16, #l1_sharded>, tensor<32x32xbf16, #l1_sharded>) -> tensor<32x32xbf16, #l1_sharded>
    return %result : tensor<32x32xbf16, #l1_sharded>
  }
}

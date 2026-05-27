// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Linear + add fixture where the weight, bias, and add operand are split
// across the n300 mesh along the output feature dim. Input is replicated.
//   host input  : <10x10>            -> per-chip <10x10>  (replicated)
//   host weight : <10x20>            -> per-chip <10x10>  (sharded along dim 1)
//   host bias   : <20>               -> per-chip <10>     (sharded along dim 0)
//   host extra  : <10x20>            -> per-chip <10x10>  (sharded along dim 1)
//   per-chip linear : <10x10> @ <10x10> + <10> -> <10x10>
//   per-chip add    : <10x10> + <10x10>        -> <10x10>
//   host output : <10x20>            (aggregated along dim 1)
// Each chip computes its own slice of the output, so the per-shard
// intermediate linear tensor differs between chips.
module @Model attributes {} {
  func.func @forward(
      %arg0: tensor<10x10xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"},
      %arg1: tensor<10x20xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "l.weight"},
      %arg2: tensor<20xf32>    {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "l.bias"},
      %arg3: tensor<10x20xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "extra_tensor"})
      -> (tensor<10x20xf32> {ttir.name = "Model.output_add_2"}) {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<10x10xf32>) -> tensor<10x10xf32>
    %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<10x20xf32>) -> tensor<10x10xf32>
    %2 = "ttir.mesh_shard"(%arg2) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<20xf32>) -> tensor<10xf32>
    %3 = "ttir.mesh_shard"(%arg3) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<10x20xf32>) -> tensor<10x10xf32>
    %4 = "ttir.linear"(%0, %1, %2) : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
    %5 = "ttir.add"(%4, %3) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %6 = "ttir.mesh_shard"(%5) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<10x10xf32>) -> tensor<10x20xf32>
    return %6 : tensor<10x20xf32>
  }
}

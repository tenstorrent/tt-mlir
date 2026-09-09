// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Linear + add fixture where all inputs (input/weight/bias/extra) are
// replicated across both chips of an n300. Each chip sees identical data and
// therefore produces identical per-shard intermediate and final tensors.
module @Model attributes {} {
  func.func @forward(
      %arg0: tensor<10x10xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"},
      %arg1: tensor<10x10xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "l.weight"},
      %arg2: tensor<10xf32>    {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "l.bias"},
      %arg3: tensor<10x10xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "extra_tensor"})
      -> (tensor<10x10xf32> {ttir.name = "Model.output_add_2"}) {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<10x10xf32>) -> tensor<10x10xf32>
    %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<10x10xf32>) -> tensor<10x10xf32>
    %2 = "ttir.mesh_shard"(%arg2) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<10xf32>) -> tensor<10xf32>
    %3 = "ttir.mesh_shard"(%arg3) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<10x10xf32>) -> tensor<10x10xf32>
    %4 = "ttir.linear"(%0, %1, %2) : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
    %5 = "ttir.add"(%4, %3) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %6 = "ttir.mesh_shard"(%5) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<10x10xf32>) -> tensor<10x10xf32>
    return %6 : tensor<10x10xf32>
  }
}

// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-layout  --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module @SyncTensorsGraph.30 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<f32>) -> (tensor<f32>) {
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: -> tensor<1xf32
    // CHECK: "ttnn.mesh_shard"
    // CHECK-SAME: -> tensor<1xf32
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: -> tensor<f32
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<f32>) -> tensor<f32>
    return %0: tensor<f32>
  }
}

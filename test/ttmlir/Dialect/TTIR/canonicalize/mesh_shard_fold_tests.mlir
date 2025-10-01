// RUN: ttmlir-opt -canonicalize %s | FileCheck %s

#ttcore_shard_direction = #ttcore.shard_direction<full_to_shard>
#ttcore_shard_type = #ttcore.shard_type<devices>

module {
  func.func @mesh_shard_1x1_fold(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func @mesh_shard_1x1_fold
    // CHECK-NOT: ttir.mesh_shard
    // CHECK: return %arg0
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.mesh_shard"(%arg0) <{
        shard_type = #ttcore_shard_type,
        shard_direction = #ttcore_shard_direction,
        shard_shape = array<i64: 1, 1>,
        shard_dims = array<i64: 0, 1>
    }> : (tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }

  func.func @mesh_shard_2x2_no_fold(%arg0: tensor<64x128xf32>) -> tensor<32x64xf32> {
    // CHECK-LABEL: func.func @mesh_shard_2x2_no_fold
    // CHECK: ttir.mesh_shard
    %0 = ttir.empty() : tensor<32x64xf32>
    %1 = "ttir.mesh_shard"(%arg0) <{
        shard_type = #ttcore_shard_type,
        shard_direction = #ttcore_shard_direction,
        shard_shape = array<i64: 2, 2>,
        shard_dims = array<i64: 0, 1>
    }> : (tensor<64x128xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }

  func.func @mesh_shard_1x2_no_fold(%arg0: tensor<64x128xf32>) -> tensor<64x64xf32> {
    // CHECK-LABEL: func.func @mesh_shard_1x2_no_fold
    // CHECK: ttir.mesh_shard
    %0 = ttir.empty() : tensor<64x64xf32>
    %1 = "ttir.mesh_shard"(%arg0) <{
        shard_type = #ttcore_shard_type,
        shard_direction = #ttcore_shard_direction,
        shard_shape = array<i64: 1, 2>,
        shard_dims = array<i64: 0, 1>
    }> : (tensor<64x128xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @mesh_shard_2x1_no_fold(%arg0: tensor<64x128xf32>) -> tensor<32x128xf32> {
    // CHECK-LABEL: func.func @mesh_shard_2x1_no_fold
    // CHECK: ttir.mesh_shard
    %0 = ttir.empty() : tensor<32x128xf32>
    %1 = "ttir.mesh_shard"(%arg0) <{
        shard_type = #ttcore_shard_type,
        shard_direction = #ttcore_shard_direction,
        shard_shape = array<i64: 2, 1>,
        shard_dims = array<i64: 0, 1>
    }> : (tensor<64x128xf32>) -> tensor<32x128xf32>
    return %1 : tensor<32x128xf32>
  }

  func.func @mesh_shard_1d_fold(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    // CHECK-LABEL: func.func @mesh_shard_1d_fold
    // CHECK-NOT: ttir.mesh_shard
    // CHECK: return %arg0
    %0 = ttir.empty() : tensor<128xf32>
    %1 = "ttir.mesh_shard"(%arg0) <{
        shard_type = #ttcore_shard_type,
        shard_direction = #ttcore_shard_direction,
        shard_shape = array<i64: 1>,
        shard_dims = array<i64: 0, 1>
    }> : (tensor<128xf32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }

  func.func @mesh_shard_3d_fold(%arg0: tensor<64x128x32xf32>) -> tensor<64x128x32xf32> {
    // CHECK-LABEL: func.func @mesh_shard_3d_fold
    // CHECK-NOT: ttir.mesh_shard
    // CHECK: return %arg0
    %0 = ttir.empty() : tensor<64x128x32xf32>
    %1 = "ttir.mesh_shard"(%arg0) <{
        shard_type = #ttcore_shard_type,
        shard_direction = #ttcore_shard_direction,
        shard_shape = array<i64: 1, 1, 1>,
        shard_dims = array<i64: 0, 1>
    }> : (tensor<64x128x32xf32>) -> tensor<64x128x32xf32>
    return %1 : tensor<64x128x32xf32>
  }

  func.func @mesh_shard_3d_no_fold(%arg0: tensor<64x128x32xf32>) -> tensor<32x128x32xf32> {
    // CHECK-LABEL: func.func @mesh_shard_3d_no_fold
    // CHECK: ttir.mesh_shard
    %0 = ttir.empty() : tensor<32x128x32xf32>
    %1 = "ttir.mesh_shard"(%arg0) <{
        shard_type = #ttcore_shard_type,
        shard_direction = #ttcore_shard_direction,
        shard_shape = array<i64: 2, 1, 1>,
        shard_dims = array<i64: 0, 1>
    }> : (tensor<64x128x32xf32>) -> tensor<32x128x32xf32>
    return %1 : tensor<32x128x32xf32>
  }

  func.func @mesh_shard_replicate_fold(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func @mesh_shard_replicate_fold
    // CHECK-NOT: ttir.mesh_shard
    // CHECK: return %arg0
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.mesh_shard"(%arg0) <{
        shard_type = #ttcore.shard_type<replicate>,
        shard_direction = #ttcore_shard_direction,
        shard_shape = array<i64: 1>,
        shard_dims = array<i64: 0, 1>
    }> : (tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}

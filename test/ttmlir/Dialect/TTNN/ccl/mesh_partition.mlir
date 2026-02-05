// // RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mesh-shape=2,4" -o %t %s
// // RUN: FileCheck %s --input-file=%t

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  ttcore.device_module {
    builtin.module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
      func.func @test_sdy_all_slice_composite_row_major(%arg0: tensor<4x32xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<4x32xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4x32xbf16>) -> tensor<4x32xbf16>
        %1 = "ttir.mesh_partition"(%0) <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<4x32xbf16>) -> tensor<1x32xbf16>
        %2 = "ttir.mesh_partition"(%1) <{cluster_axis = 0 : ui32, dim = 1 : si32}> : (tensor<1x32xbf16>) -> tensor<1x16xbf16>
        %3 = "ttir.mesh_shard"(%2) <{shard_dims = array<i64: 1, 0>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 4, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x16xbf16>) -> tensor<4x32xbf16>
        return %3 : tensor<4x32xbf16>
      }
    }
  }
}

// CHECK-LABEL: @test_sdy_all_slice_composite_row_major
// CHECK-NOT: "ttnn.to_layout"
// CHECK: "ttnn.mesh_partition"(%arg0)
// CHECK-SAME: <{cluster_axis = 1 : ui32, dim = 0 : si32}>
// CHECK: "ttnn.mesh_partition"(%1)
// CHECK-SAME: <{cluster_axis = 0 : ui32, dim = 1 : si32}>

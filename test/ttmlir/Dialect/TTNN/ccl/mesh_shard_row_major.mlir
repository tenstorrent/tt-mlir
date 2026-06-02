// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that conv3d output layout (ROW_MAJOR) is preserved through an identity
// mesh_shard. The identity variant is a no-op at runtime and cannot perform
// implicit tilization, so its output layout must match its input layout.
//
// Capture the ROW_MAJOR alias for the mesh_shard result (1x12x26x26x16,
// collapsed to memref<8112x16xbf16> where 8112 = 1*12*26*26).
//
// CHECK: [[RM_LAYOUT:#ttnn_layout[0-9]+]] = #ttnn.ttnn_layout<{{.*}}memref<8112x16xbf16, #dram>

module @conv3d_identity_mesh_shard_layout attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @forward(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<16x4x3x3x3xbf16>) -> tensor<1x12x26x26x16xbf16> {
    // CHECK: "ttnn.conv3d"
    %0 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              groups = 1 : i32,
              padding_mode = "zeros"
            }> : (tensor<1x8x28x28x4xbf16>, tensor<16x4x3x3x3xbf16>) -> tensor<1x6x26x26x16xbf16>
    // CHECK: "ttnn.mesh_shard"([[ARG:.*]], [[DEV:.*]]) <{
    // CHECK-SAME: shard_type = #ttcore.shard_type<identity>
    // CHECK-SAME: -> tensor<1x12x26x26x16xbf16, [[RM_LAYOUT]]>
    %1 = "ttir.mesh_shard"(%0)
            <{
              shard_dims = array<i64: -1, 1>,
              shard_direction = #ttcore.shard_direction<shard_to_full>,
              shard_shape = array<i64: 1, 2, 1, 1, 1>,
              shard_type = #ttcore.shard_type<identity>
            }> : (tensor<1x6x26x26x16xbf16>) -> tensor<1x12x26x26x16xbf16>
    return %1 : tensor<1x12x26x26x16xbf16>
  }
}

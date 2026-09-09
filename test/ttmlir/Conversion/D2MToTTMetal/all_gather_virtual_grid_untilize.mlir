// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="mock-system-desc-arch=wormhole_b0 mesh-shape=1,8 mesh-topology=linear,ring" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x8>]>} {
  // CHECK-LABEL: func.func @all_gather_virtual_grid_untilize
  // CHECK: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64, virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}}> : () -> memref<16x1x128x256xf32
  // CHECK: #ttmetal.core_range<0x0, 4x4>
  // The virtual 16x1 grid is executed on its 4x4 physical core range.
  // CHECK: #ttmetal.core_range<0x0, 4x4>
  func.func @all_gather_virtual_grid_untilize(%arg0: tensor<256x2048xf32>) -> tensor<2048x2048xf32> {
    %0 = "ttir.mesh_shard"(%arg0) <{
      shard_dims = array<i64: 0, 1>,
      shard_direction = #ttcore.shard_direction<full_to_shard>,
      shard_shape = array<i64: 1, 8>,
      shard_type = #ttcore.shard_type<devices>
    }> : (tensor<256x2048xf32>) -> tensor<256x256xf32>
    %1 = "ttir.all_gather"(%0) <{
      all_gather_dim = 0 : si32,
      cluster_axis = 1 : ui32
    }> : (tensor<256x256xf32>) -> tensor<2048x256xf32>
    %2 = "ttir.mesh_shard"(%1) <{
      shard_dims = array<i64: 0, 1>,
      shard_direction = #ttcore.shard_direction<shard_to_full>,
      shard_shape = array<i64: 1, 8>,
      shard_type = #ttcore.shard_type<devices>
    }> : (tensor<2048x256xf32>) -> tensor<2048x2048xf32>
    return %2 : tensor<2048x2048xf32>
  }
}

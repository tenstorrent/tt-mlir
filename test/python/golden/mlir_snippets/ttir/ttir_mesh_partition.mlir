module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  func.func @mesh_partition(%arg0: tensor<4x32xbf16>) -> tensor<1x32xbf16> {
    %0 = "ttir.mesh_partition"(%arg0) <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<4x32xbf16>) -> tensor<1x32xbf16>
    return %0 : tensor<1x32xbf16>
  }
}

module {
  func.func @mesh_partition(%arg0: tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32> {
    %0 = "ttir.mesh_partition"(%arg0) <{cluster_axis = 1 : ui32, dim = 2 : si32}> : (tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32>
    return %0 : tensor<4x4x128x128xf32>
  }
}

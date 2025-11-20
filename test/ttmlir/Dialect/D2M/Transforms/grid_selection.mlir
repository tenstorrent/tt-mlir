module attributes {ttcore.device = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>} {
  func.func @test_grid_selection(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = "ttir.exp"(%arg0, %arg0) : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}

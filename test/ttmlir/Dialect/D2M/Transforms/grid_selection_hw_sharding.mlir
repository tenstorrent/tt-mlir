// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --canonicalize %s | FileCheck %s

#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

module attributes {ttcore.device = #any_device} {
  func.func @test_height_sharded_virtual_grid(%arg0: tensor<2048x32xf32>) -> tensor<2048x32xf32> {
    // CHECK-LABEL: func.func @test_height_sharded_virtual_grid
    // CHECK: d2m.empty() {{.*}} : tensor<64x1x1x1x!ttcore.tile<32x32, f32>
    // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<64x1,
     %0 = "ttir.exp"(%arg0) : (tensor<2048x32xf32>) -> tensor<2048x32xf32>
    return %0 : tensor<2048x32xf32>
  }
}

// -----

#any_device_2 = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

module attributes {ttcore.device = #any_device_2} {
  func.func @test_width_sharded_virtual_grid(%arg0: tensor<32x2048xf32>) -> tensor<32x2048xf32> {
    // CHECK-LABEL: func.func @test_width_sharded_virtual_grid
    // CHECK: d2m.empty() {{.*}} : tensor<1x64x1x1x!ttcore.tile<32x32, f32>
    // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<1x64,
     %0 = "ttir.exp"(%arg0) : (tensor<32x2048xf32>) -> tensor<32x2048xf32>
    return %0 : tensor<32x2048xf32>
  }
}

// -----

#any_device_3 = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

module attributes {ttcore.device = #any_device_3} {
  func.func @test_width_sharded_sub64(%arg0: tensor<32x1280xf32>) -> tensor<32x1280xf32> {
    // Physical shape in tiles: 1x40. Block sharding gives 1x8 = 8 cores.
    // Virtual grid should select 1x40, packed into 5x8 physical grid.
    // CHECK-LABEL: func.func @test_width_sharded_sub64
    // CHECK: d2m.empty() {{.*}} : tensor<1x40x1x1x!ttcore.tile<32x32, f32>
    // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<1x40,
     %0 = "ttir.exp"(%arg0) : (tensor<32x1280xf32>) -> tensor<32x1280xf32>
    return %0 : tensor<32x1280xf32>
  }
}

// -----

#any_device_2 = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

module attributes {ttcore.device = #any_device_2} {
  func.func @test_height_sharded_sub64(%arg0: tensor<1536x32xf32>) -> tensor<1536x32xf32> {
    // Physical shape in tiles: 48x1. Block sharding gives 8x1 = 8 cores.
    // Virtual grid should select 48x1, packed into 6x8 physical grid.
    // CHECK-LABEL: func.func @test_height_sharded_sub64
    // CHECK: d2m.empty() {{.*}} : tensor<48x1x1x1x!ttcore.tile<32x32, f32>
    // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<48x1,
     %0 = "ttir.exp"(%arg0) : (tensor<1536x32xf32>) -> tensor<1536x32xf32>
    return %0 : tensor<1536x32xf32>
  }
}

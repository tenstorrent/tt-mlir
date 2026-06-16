// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --canonicalize %s | FileCheck %s

#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

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

#any_device_2 = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

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

#any_device_3 = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

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

#any_device_2 = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

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

// -----

#any_device_4 = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

module attributes {ttcore.device = #any_device_4} {
  func.func @test_2d_factor_product_virtual_grid(%arg0: tensor<96x1024xf32>) -> tensor<96x1024xf32> {
    // Physical shape in tiles: 3x32. Block sharding gives 3x8 = 24 cores,
    // so main's existing virtual-grid gate already wants VGM; broadening only
    // changes the realized virtual grid from single-axis 1x32 to 3x16.
    // CHECK-LABEL: func.func @test_2d_factor_product_virtual_grid
    // CHECK: d2m.empty() {{.*}} : tensor<3x16x1x2x!ttcore.tile<32x32, f32>
    // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<3x16,
    %0 = "ttir.exp"(%arg0) : (tensor<96x1024xf32>) -> tensor<96x1024xf32>
    return %0 : tensor<96x1024xf32>
  }
}

// -----

#any_device_5 = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

module attributes {ttcore.device = #any_device_5} {
  func.func @test_matmul_keeps_single_axis_virtual_grid(%lhs: tensor<96x1024xf32>, %rhs: tensor<1024x512xf32>) -> tensor<96x512xf32> {
    // Matmul keeps main's single-axis virtual-grid behavior for source layouts;
    // the compute generic remains physically placed.
    // CHECK-LABEL: func.func @test_matmul_keeps_single_axis_virtual_grid
    // CHECK: d2m.empty() {{.*}} : tensor<1x32x3x1x!ttcore.tile<32x32, f32>
    // CHECK-NOT: grid = #ttcore.grid<3x16
    // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<3x8>
    %0 = "ttir.matmul"(%lhs, %rhs) : (tensor<96x1024xf32>, tensor<1024x512xf32>) -> tensor<96x512xf32>
    return %0 : tensor<96x512xf32>
  }
}

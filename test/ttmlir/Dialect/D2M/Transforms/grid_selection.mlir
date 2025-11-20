// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --split-input-file %s | FileCheck %s --check-prefix=CHECK-BEFORE
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --split-input-file %s | FileCheck %s --check-prefix=CHECK-AFTER
#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

module attributes {ttcore.device = #any_device} {
  func.func @test_grid_selection(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    // CHECK-BEFORE-LABEL: func.func @test_grid_selection
    // Verify TTIRToD2M creates 1x1 grids
    // CHECK-BEFORE: d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>
    // CHECK-BEFORE: d2m.to_layout %arg0, %{{.*}} : tensor<256x256xf32> into tensor<1x1x8x8x!ttcore.tile<32x32, f32>
    // CHECK-BEFORE: d2m.generic {{{.*}}grid = #ttcore.grid<1x1>

    // CHECK-AFTER-LABEL: func.func @test_grid_selection
    // Verify D2MGridSelection optimizes to 8x8 grids
    // CHECK-AFTER: d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.to_layout %arg0, %{{.*}} : tensor<256x256xf32> into tensor<8x8x1x1x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>

    %0 = "ttir.exp"(%arg0, %arg0) : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}

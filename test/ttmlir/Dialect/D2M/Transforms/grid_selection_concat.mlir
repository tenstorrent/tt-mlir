// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns --d2m-grid-selection --canonicalize %s | FileCheck %s

// Verify that grid selection propagates multi-core grids to upstream to_layout
// ops for concat inputs. Without this, large concat inputs remain on a 1x1
// grid and overflow L1 when multiple buffers must coexist.

#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

// Test: Large 4D concat along dim 3 (the shape that triggered L1 overflow).
// Each input is 1x32x128x64 bf16 = 524,288 bytes. On a 1x1x1x1 grid, three
// concurrent buffers would need ~1.5MB, exceeding ~1.4MB usable L1.
// Grid selection should distribute the to_layout outputs across multiple cores.
module attributes {ttcore.device = #any_device} {
  func.func @concat_large_4d(%arg0: tensor<1x32x128x64xbf16>, %arg1: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x128xbf16> {
    // CHECK-LABEL: func.func @concat_large_4d
    //
    // Verify to_layout ops use a multi-core grid (1x8x4x2), NOT 1x1x1x1.
    // CHECK: d2m.to_layout %arg0, %{{.*}} : tensor<1x32x128x64xbf16> into tensor<1x8x4x2x
    // CHECK: d2m.to_layout %arg1, %{{.*}} : tensor<1x32x128x64xbf16> into tensor<1x8x4x2x
    //
    // Verify composite_view and generic still exist with a multi-core grid.
    // CHECK: d2m.composite_view
    // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<1x4x4x4
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 3 : si32}> : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x128xbf16>
    return %0 : tensor<1x32x128x128xbf16>
  }
}

// -----

#any_device_2 = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

// Test: 2D concat where inputs are large enough to benefit from multi-core.
module attributes {ttcore.device = #any_device_2} {
  func.func @concat_large_2d(%arg0: tensor<256x256xbf16>, %arg1: tensor<256x256xbf16>) -> tensor<256x512xbf16> {
    // CHECK-LABEL: func.func @concat_large_2d
    // CHECK: d2m.composite_view
    // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 1 : si32}> : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x512xbf16>
    return %0 : tensor<256x512xbf16>
  }
}

// -----

#any_device_3 = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

// Test: 3-input concat to verify all upstream to_layout ops get updated.
module attributes {ttcore.device = #any_device_3} {
  func.func @concat_three_inputs(%arg0: tensor<1x32x128x64xbf16>, %arg1: tensor<1x32x128x64xbf16>, %arg2: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x192xbf16> {
    // CHECK-LABEL: func.func @concat_three_inputs
    // All three to_layout ops should use multi-core grids (not 1x1x1x1).
    // CHECK: d2m.to_layout %arg0, %{{.*}} : tensor<1x32x128x64xbf16> into tensor<1x8x4x2x
    // CHECK: d2m.to_layout %arg1, %{{.*}} : tensor<1x32x128x64xbf16> into tensor<1x8x4x2x
    // CHECK: d2m.to_layout %arg2, %{{.*}} : tensor<1x32x128x64xbf16> into tensor<1x8x4x2x
    // CHECK: d2m.composite_view
    // CHECK: d2m.generic
    %0 = "ttir.concat"(%arg0, %arg1, %arg2) <{dim = 3 : si32}> : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x192xbf16>
    return %0 : tensor<1x32x128x192xbf16>
  }
}

// -----

#any_device_4 = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

module attributes {ttcore.device = #any_device_4} {
  func.func @concat_row_major(%arg0: tensor<136x448xf32>, %arg1: tensor<136x448xf32>) -> tensor<272x448xf32> {
    // CHECK-LABEL: func.func @concat_row_major
    // CHECK: d2m.to_layout %arg0, %{{.*}} : tensor<136x448xf32> into tensor<5x8x32x64xf32
    // CHECK: d2m.to_layout %arg1, %{{.*}} : tensor<136x448xf32> into tensor<5x8x32x64xf32
    // CHECK: d2m.composite_view
    // CHECK: d2m.generic
    // CHECK: d2m.to_layout %{{.*}}, %{{.*}} : tensor<8x8x64x64xf32{{.*}} into tensor<272x448xf32>
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 0 : si32}> : (tensor<136x448xf32>, tensor<136x448xf32>) -> tensor<272x448xf32>
    return %0 : tensor<272x448xf32>
  }
}

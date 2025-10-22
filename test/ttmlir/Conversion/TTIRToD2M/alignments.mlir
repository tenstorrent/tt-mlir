// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection %s | FileCheck %s

// Simple check to ensure we get valid metal_layout for each size.
// We want to ensure dim_alignments round up to 256 (on 8x8) if the physical shape is > 8 after tilizing.
// Otherwise it should just be tile size (32).

// CHECK-DAG: #[[LAYOUT_SMALL:.*]] = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_MEDIUM:.*]] = #ttcore.metal_layout<logical_shape = 128x96, dim_alignments = 32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_LARGE:.*]] = #ttcore.metal_layout<logical_shape = 512x512, dim_alignments = 256x256, {{.*}}>
// CHECK-DAG: #[[LAYOUT_LARGE_H:.*]] = #ttcore.metal_layout<logical_shape = 512x128, dim_alignments = 256x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_LARGE_W:.*]] = #ttcore.metal_layout<logical_shape = 128x512, dim_alignments = 32x256, {{.*}}>
// CHECK-DAG: #[[LAYOUT_3D:.*]] = #ttcore.metal_layout<logical_shape = 2x128x96, dim_alignments = 1x32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_4D:.*]] = #ttcore.metal_layout<logical_shape = 2x2x64x64, dim_alignments = 1x1x32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_BOUNDARY:.*]] = #ttcore.metal_layout<logical_shape = 256x32, dim_alignments = 32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_UNDER:.*]] = #ttcore.metal_layout<logical_shape = 255x32, dim_alignments = 32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_ABOVE:.*]] = #ttcore.metal_layout<logical_shape = 257x32, dim_alignments = 256x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_NONALIGNED_2D:.*]] = #ttcore.metal_layout<logical_shape = 100x100, dim_alignments = 32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_NONALIGNED_3D_LARGE_H:.*]] = #ttcore.metal_layout<logical_shape = 5x37x11, dim_alignments = 256x32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_NONALIGNED_3D_LARGE_W:.*]] = #ttcore.metal_layout<logical_shape = 3x61x419, dim_alignments = 1x32x256, {{.*}}>
// CHECK-DAG: #[[LAYOUT_NONALIGNED_4D:.*]] = #ttcore.metal_layout<logical_shape = 1x19x1x1, dim_alignments = 256x1x32x32, {{.*}}>

module {
  // CHECK-LABEL: func @test_alignment_rules
  func.func @test_alignment_rules(
    %small: tensor<64x64xf32>,
    %medium: tensor<128x96xf32>,
    %large: tensor<512x512xf32>,
    %large_h: tensor<512x128xf32>,
    %large_w: tensor<128x512xf32>,
    %tensor_3d: tensor<2x128x96xf32>,
    %tensor_4d: tensor<2x2x64x64xf32>
  ) -> tensor<64x64xf32> {
    %0 = "ttir.add"(%small, %small, %small) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %1 = "ttir.add"(%medium, %medium, %medium) : (tensor<128x96xf32>, tensor<128x96xf32>, tensor<128x96xf32>) -> tensor<128x96xf32>
    %2 = "ttir.add"(%large, %large, %large) : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    %3 = "ttir.add"(%large_h, %large_h, %large_h) : (tensor<512x128xf32>, tensor<512x128xf32>, tensor<512x128xf32>) -> tensor<512x128xf32>
    %4 = "ttir.add"(%large_w, %large_w, %large_w) : (tensor<128x512xf32>, tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %5 = "ttir.add"(%tensor_3d, %tensor_3d, %tensor_3d) : (tensor<2x128x96xf32>, tensor<2x128x96xf32>, tensor<2x128x96xf32>) -> tensor<2x128x96xf32>
    %6 = "ttir.add"(%tensor_4d, %tensor_4d, %tensor_4d) : (tensor<2x2x64x64xf32>, tensor<2x2x64x64xf32>, tensor<2x2x64x64xf32>) -> tensor<2x2x64x64xf32>
    return %0 : tensor<64x64xf32>
  }

  // CHECK-LABEL: func @test_edge_cases
  func.func @test_edge_cases(
    %boundary: tensor<256x32xf32>,
    %under_boundary: tensor<255x32xf32>,
    %above_boundary: tensor<257x32xf32>,
    %non_aligned_2d: tensor<100x100xf32>,
    %non_aligned_3d_large_h : tensor<5x37x11xf32>,
    %non_aligned_3d_large_w : tensor<3x61x419xf32>,
    %non_aligned_4d : tensor<1x19x1x1xf32>
  ) -> tensor<256x32xf32> {
    %0 = "ttir.exp"(%boundary, %boundary) : (tensor<256x32xf32>, tensor<256x32xf32>) -> tensor<256x32xf32>
    %1 = "ttir.exp"(%under_boundary, %under_boundary) : (tensor<255x32xf32>, tensor<255x32xf32>) -> tensor<255x32xf32>
    %2 = "ttir.exp"(%above_boundary, %above_boundary) : (tensor<257x32xf32>, tensor<257x32xf32>) -> tensor<257x32xf32>
    %3 = "ttir.exp"(%non_aligned_2d, %non_aligned_2d) : (tensor<100x100xf32>, tensor<100x100xf32>) -> tensor<100x100xf32>
    %4 = "ttir.exp"(%non_aligned_3d_large_h, %non_aligned_3d_large_h) : (tensor<5x37x11xf32>, tensor<5x37x11xf32>) -> tensor<5x37x11xf32>
    %5 = "ttir.exp"(%non_aligned_3d_large_w, %non_aligned_3d_large_w) : (tensor<3x61x419xf32>, tensor<3x61x419xf32>) -> tensor<3x61x419xf32>
    %6 = "ttir.exp"(%non_aligned_4d, %non_aligned_4d) : (tensor<1x19x1x1xf32>, tensor<1x19x1x1xf32>) -> tensor<1x19x1x1xf32>
    return %0 : tensor<256x32xf32>
  }
}

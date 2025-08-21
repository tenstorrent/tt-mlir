// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttir-generic %s | FileCheck %s

// Simple check to ensure we get valid metal_layout for each size;
// we want to ensure dim_alignments round up to 256 (on 8x8) if the physical shape is >= 8 after tilizing.
// Otherwise it should just be tile size (32).

// CHECK-DAG: #[[LAYOUT_SMALL:.*]] = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_MEDIUM:.*]] = #ttcore.metal_layout<logical_shape = 128x96, dim_alignments = 32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_LARGE:.*]] = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_MIXED:.*]] = #ttcore.metal_layout<logical_shape = 512x128, dim_alignments = 256x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_3D:.*]] = #ttcore.metal_layout<logical_shape = 2x128x96, dim_alignments = 1x32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_4D:.*]] = #ttcore.metal_layout<logical_shape = 2x2x64x64, dim_alignments = 1x1x32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_BOUNDARY:.*]] = #ttcore.metal_layout<logical_shape = 256x32, dim_alignments = 32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_UNDER:.*]] = #ttcore.metal_layout<logical_shape = 255x32, dim_alignments = 32x32, {{.*}}>
// CHECK-DAG: #[[LAYOUT_NONALIGNED:.*]] = #ttcore.metal_layout<logical_shape = 100x100, dim_alignments = 32x32, {{.*}}>

module {
  // CHECK-LABEL: func @test_alignment_rules
  func.func @test_alignment_rules(
    %small: tensor<64x64xf32>,
    %medium: tensor<128x96xf32>,
    %large: tensor<256x256xf32>,
    %mixed: tensor<512x128xf32>,
    %tensor_3d: tensor<2x128x96xf32>,
    %tensor_4d: tensor<2x2x64x64xf32>
  ) -> tensor<256x256xf32> {
    %0 = "ttir.add"(%small, %small, %small) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %1 = "ttir.add"(%medium, %medium, %medium) : (tensor<128x96xf32>, tensor<128x96xf32>, tensor<128x96xf32>) -> tensor<128x96xf32>
    %2 = "ttir.add"(%large, %large, %large) : (tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %3 = "ttir.add"(%mixed, %mixed, %mixed) : (tensor<512x128xf32>, tensor<512x128xf32>, tensor<512x128xf32>) -> tensor<512x128xf32>
    %4 = "ttir.add"(%tensor_3d, %tensor_3d, %tensor_3d) : (tensor<2x128x96xf32>, tensor<2x128x96xf32>, tensor<2x128x96xf32>) -> tensor<2x128x96xf32>
    %5 = "ttir.add"(%tensor_4d, %tensor_4d, %tensor_4d) : (tensor<2x2x64x64xf32>, tensor<2x2x64x64xf32>, tensor<2x2x64x64xf32>) -> tensor<2x2x64x64xf32>
    return %2 : tensor<256x256xf32>

  }

  // CHECK-LABEL: func @test_edge_cases
  func.func @test_edge_cases(
    %boundary: tensor<256x32xf32>,
    %under_boundary: tensor<255x32xf32>,
    %non_aligned: tensor<100x100xf32>
  ) -> tensor<256x32xf32> {
    %0 = "ttir.exp"(%boundary, %boundary) : (tensor<256x32xf32>, tensor<256x32xf32>) -> tensor<256x32xf32>
    %1 = "ttir.exp"(%under_boundary, %under_boundary) : (tensor<255x32xf32>, tensor<255x32xf32>) -> tensor<255x32xf32>
    %2 = "ttir.exp"(%non_aligned, %non_aligned) : (tensor<100x100xf32>, tensor<100x100xf32>) -> tensor<100x100xf32>
    return %0 : tensor<256x32xf32>
  }
}

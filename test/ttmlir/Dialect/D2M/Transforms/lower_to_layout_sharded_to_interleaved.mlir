// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_sharded = #ttcore.metal_layout<logical_shape = 1x1x32x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 3], [3, 4]]> : tensor<2x2xi64>, undef, l1, sharded>
#dram_interleaved = #ttcore.metal_layout<logical_shape = 1x1x32x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 3], [3, 4]]> : tensor<2x2xi64>, undef, dram, interleaved>

// CHECK-LABEL: func.func @sharded_to_interleaved
func.func @sharded_to_interleaved() -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #dram_interleaved> {
  %src = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, 0, 0)>} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1_sharded>
  %dst = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #dram_interleaved>

  // CHECK: %[[SRC:.*]] = d2m.empty() {virtualGridForwardMapping = #map{{.*}}, virtualGridInverseMapping = #map{{.*}}} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[DST:.*]] = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[RESULT:.*]] = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK-NEXT: ins(%[[SRC]] : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-NEXT: outs(%[[DST]] : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: d2m.remote_load {{.*}} %[[SRC]][%{{.*}}, %{{.*}}] : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: d2m.remote_store %[[DST]][%{{.*}}, %{{.*}}] {{.*}} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: return %[[RESULT]]

  %1 = d2m.to_layout %src, %dst : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1_sharded> into tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #dram_interleaved>
    -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #dram_interleaved>

  return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #dram_interleaved>
}

#l1_sharded_2 = #ttcore.metal_layout<logical_shape = 32x2048, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#dram_interleaved_2 = #ttcore.metal_layout<logical_shape = 32x2048, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, interleaved>

#ttnn_dram_interleaved_2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x64x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>, exactGrid = true>

// CHECK-LABEL: func.func @sharded_to_interleaved_reblock
func.func @sharded_to_interleaved_reblock() -> tensor<32x2048xbf16, #ttnn_dram_interleaved_2> {
  %src = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> ((d1 floordiv 8) mod 8, d1 mod 8, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, 0, (d1 + d0 * 8) mod 64)>} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #l1_sharded_2>
  %dst = d2m.empty() : tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #dram_interleaved_2>

  // CHECK: %[[SRC:.*]] = d2m.empty() {virtualGridForwardMapping = #map{{.*}}, virtualGridInverseMapping = #map{{.*}}} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[DST:.*]] = d2m.empty() : tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[DST_VIEW:.*]] = d2m.view_layout %[[DST]] remapping = #map{{.*}} : tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}> -> tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[RESULT:.*]] = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x64
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK-NEXT: ins(%[[SRC]] : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-NEXT: outs(%[[DST_VIEW]] : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: d2m.remote_load {{.*}} %[[SRC]][%{{.*}}, %{{.*}}] : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: d2m.remote_store %[[DST_VIEW]][%{{.*}}, %{{.*}}] {{.*}} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
// CHECK: %[[RESULT_VIEW:.*]] = d2m.view_layout %[[RESULT]] remapping = #map{{.*}} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}> -> tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
// CHECK: ttir.ttnn_metal_layout_cast %[[RESULT_VIEW]]

  %1 = d2m.to_layout %src, %dst : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #l1_sharded_2> into tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #dram_interleaved_2>
    -> tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #dram_interleaved_2>

  %cast = ttir.ttnn_metal_layout_cast %1 : tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #dram_interleaved_2> -> tensor<32x2048xbf16, #ttnn_dram_interleaved_2>
  return %cast : tensor<32x2048xbf16, #ttnn_dram_interleaved_2>
}

#dram_sharded_3 = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, sharded>
#dram_interleaved_3 = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, interleaved>

// CHECK-LABEL: func.func @dram_sharded_to_interleaved
func.func @dram_sharded_to_interleaved() -> tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #dram_interleaved_3> {
  %src = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #dram_sharded_3>
  %dst = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #dram_interleaved_3>

  // CHECK: %[[SRC:.*]] = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[DST:.*]] = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[DST_VIEW:.*]] = d2m.view_layout %[[DST]] remapping = #map{{.*}} : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}> -> tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[RESULT:.*]] = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK-NEXT: ins(%[[SRC]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-NEXT: outs(%[[DST_VIEW]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: d2m.remote_load {{.*}} %[[SRC]][%{{.*}}, %{{.*}}] : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: d2m.remote_store %[[DST_VIEW]][%{{.*}}, %{{.*}}] {{.*}} : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[RESULT_VIEW:.*]] = d2m.view_layout %[[RESULT]] remapping = #map{{.*}} : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}> -> tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: return %[[RESULT_VIEW]]

  %result = d2m.to_layout %src, %dst : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #dram_sharded_3> into tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #dram_interleaved_3>
    -> tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #dram_interleaved_3>

  return %result : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #dram_interleaved_3>
}

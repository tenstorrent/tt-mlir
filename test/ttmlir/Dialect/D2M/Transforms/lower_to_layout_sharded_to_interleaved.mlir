// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_sharded = #ttcore.metal_layout<logical_shape = 1x1x32x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 3], [3, 4]]> : tensor<2x2xi64>, undef, l1, sharded>
#dram_interleaved = #ttcore.metal_layout<logical_shape = 1x1x32x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 3], [3, 4]]> : tensor<2x2xi64>, undef, dram, interleaved>

// CHECK-LABEL: func.func @sharded_to_interleaved
func.func @sharded_to_interleaved() -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #dram_interleaved> {
  %src = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, 0, 0)>} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1_sharded>
  %dst = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #dram_interleaved>

  // CHECK: %[[SRC:.*]] = d2m.empty() {virtualGridForwardMapping = #map, virtualGridInverseMapping = #map1} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[DST:.*]] = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: %[[RESULT:.*]] = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, 0, 0), physical_to_virt_map = (d0, d1) -> (0, 0, 0)>
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK-NEXT: ins(%[[SRC]] : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout1>)
  // CHECK-NEXT: outs(%[[DST]] : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
  // CHECK: d2m.remote_load {{.*}} %[[SRC]][%{{.*}}, %{{.*}}] : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: d2m.remote_store %[[DST]][%{{.*}}, %{{.*}}] {{.*}} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK-NOT: d2m.generic
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

  // CHECK: %[[SRC:.*]] = d2m.empty() {virtualGridForwardMapping = #map3, virtualGridInverseMapping = #map4} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout2>
  // CHECK: %[[DST:.*]] = d2m.empty() : tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #layout3>
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %[[DST]] remapping = #map5 : tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #layout3> -> tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout3>
  // CHECK: %[[RESULT:.*]] = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x64, virt_to_physical_map = (d0, d1) -> (0, d1 floordiv 8, d1 mod 8), physical_to_virt_map = (d0, d1) -> (0, 0, (d1 + d0 * 8) mod 64)>
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK-NEXT: ins(%[[SRC]] : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK-NEXT: outs(%[[VIEW]] : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout3>)
  // CHECK: d2m.remote_load {{.*}} %[[SRC]][%{{.*}}, %{{.*}}] : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout2>
  // CHECK: d2m.remote_store %[[VIEW]][%{{.*}}, %{{.*}}] {{.*}} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout3>
  // CHECK: } : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout3>
  // CHECK-NOT: d2m.generic
  // CHECK: ttir.ttnn_metal_layout_cast %[[RESULT]]

  %1 = d2m.to_layout %src, %dst : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #l1_sharded_2> into tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #dram_interleaved_2>
    -> tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #dram_interleaved_2>

  %cast = ttir.ttnn_metal_layout_cast %1 : tensor<1x1x1x64x!ttcore.tile<32x32, bf16>, #dram_interleaved_2> -> tensor<32x2048xbf16, #ttnn_dram_interleaved_2>
  return %cast : tensor<32x2048xbf16, #ttnn_dram_interleaved_2>
}

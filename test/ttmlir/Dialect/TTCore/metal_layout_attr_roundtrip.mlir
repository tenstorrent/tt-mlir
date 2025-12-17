// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s

// Layouts are hoisted to top of module as aliases.
// CHECK-DAG: #{{.*}} = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = {{.*}}, undef, l1, sharded, index_map = map(0)>
// CHECK-DAG: #{{.*}} = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = {{.*}}, zero, l1, interleaved, index_map = map(0)>
// CHECK-DAG: #{{.*}} = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = {{.*}}, zero, l1, sharded, index_map = map(0)>
// CHECK-DAG: #{{.*}} = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = {{.*}}, zero, dram, interleaved, index_map = map(0)>

// CHECK-LABEL: func.func @test_sharded_memory_layout
// CHECK-LABEL: func.func @test_interleaved_memory_layout
// CHECK-LABEL: func.func @test_oob_zero_sharded
// CHECK-LABEL: func.func @test_dram_interleaved

// Test round-trip for MetalLayoutAttr with sharded memory layout
#layout_sharded = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

func.func @test_sharded_memory_layout(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_sharded>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_sharded> {
  return %arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_sharded>
}

// Test round-trip for MetalLayoutAttr with interleaved memory layout
#layout_interleaved = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, zero, l1, interleaved, index_map = map(0)>

func.func @test_interleaved_memory_layout(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>, #layout_interleaved>) -> tensor<2x2x!ttcore.tile<32x32, f32>, #layout_interleaved> {
  return %arg0 : tensor<2x2x!ttcore.tile<32x32, f32>, #layout_interleaved>
}

// Test round-trip for different OOB values with sharded layout
#layout_oob_zero = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, zero, l1, sharded, index_map = map(0)>

func.func @test_oob_zero_sharded(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>, #layout_oob_zero>) -> tensor<2x2x!ttcore.tile<32x32, f32>, #layout_oob_zero> {
  return %arg0 : tensor<2x2x!ttcore.tile<32x32, f32>, #layout_oob_zero>
}

// Test round-trip for DRAM memory space with interleaved layout
#layout_dram = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, zero, dram, interleaved, index_map = map(0)>

func.func @test_dram_interleaved(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>, #layout_dram>) -> tensor<4x4x!ttcore.tile<32x32, f32>, #layout_dram> {
  return %arg0 : tensor<4x4x!ttcore.tile<32x32, f32>, #layout_dram>
}

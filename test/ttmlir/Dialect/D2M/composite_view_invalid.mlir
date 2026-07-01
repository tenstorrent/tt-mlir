// RUN: ttmlir-opt --ttcore-register-device --verify-diagnostics --split-input-file %s

#l1 = #ttcore.memory_space<l1>

// Rule 1 = only one input.
func.func @composite_view_one_input() -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> {
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // expected-error @+1 {{must have at least two inputs.}}
  %0 = "d2m.composite_view"(%alloc_0) <{dim = 1 : si32}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
  return %0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
}

// -----

// Rule 2 = dim out of range (negative).
#l1_r2neg = #ttcore.memory_space<l1>
func.func @composite_view_dim_negative() -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_r2neg> {
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_r2neg>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_r2neg>
  // expected-error @+1 {{dim out of range.}}
  %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = -1 : si32}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_r2neg>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_r2neg>) -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_r2neg>
  return %0 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_r2neg>
}

// -----

// Rule 2 = dim out of range (>= rank).
#l1_r2pos = #ttcore.memory_space<l1>
func.func @composite_view_dim_too_large() -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_r2pos> {
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_r2pos>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_r2pos>
  // expected-error @+1 {{dim out of range.}}
  %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 2 : si32}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_r2pos>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_r2pos>) -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_r2pos>
  return %0 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_r2pos>
}

// -----

// Rule 3 = inputs/output rank mismatch.
#l1_rank = #ttcore.memory_space<l1>
func.func @composite_view_rank_mismatch() -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_rank> {
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096x4096, 1>, #l1_rank>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_rank>
  // expected-error @+1 {{incompatible inputs & output ranks.}}
  %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32}> : (memref<1x1x1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096x4096, 1>, #l1_rank>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_rank>) -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_rank>
  return %0 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_rank>
}

// -----

// Rule 4 = non-composite dim mismatch between inputs and output.
#l1_noncomp = #ttcore.memory_space<l1>
func.func @composite_view_noncomposite_dim_mismatch() -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_noncomp> {
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_noncomp>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_noncomp>
  // expected-error @+1 {{incompatible non-composite dim.}}
  %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32}> : (memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_noncomp>, memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_noncomp>) -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_noncomp>
  return %0 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_noncomp>
}

// -----

// Rule 5 = composite dim sum across inputs != output (tensor-typed).
#linput_cd = #ttcore.metal_layout<logical_shape = 32x4, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#lout_cd = #ttcore.metal_layout<logical_shape = 32x16, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
func.func @composite_view_composite_dim_mismatch() -> tensor<1x1x32x32xf32, #lout_cd> {
  %0 = d2m.empty() : tensor<1x1x32x32xf32, #linput_cd>
  %1 = d2m.empty() : tensor<1x1x32x32xf32, #linput_cd>
  // expected-error @+1 {{incompatible composite dim.}}
  %2 = "d2m.composite_view"(%0, %1) <{dim = 1 : si32}> : (tensor<1x1x32x32xf32, #linput_cd>, tensor<1x1x32x32xf32, #linput_cd>) -> tensor<1x1x32x32xf32, #lout_cd>
  return %2 : tensor<1x1x32x32xf32, #lout_cd>
}

// -----

// Rule 6 = logicalSizes attr set on tiled output.
#l1_unneeded = #ttcore.memory_space<l1>
func.func @composite_view_unneeded_logical_sizes() -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_unneeded> {
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_unneeded>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_unneeded>
  // expected-error @+1 {{unneeded logicalSizes attr.}}
  %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32, logicalSizes = array<i64: 32, 32>}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_unneeded>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_unneeded>) -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_unneeded>
  return %0 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_unneeded>
}

// -----

// Rule 7 = logicalSizes missing on row-major output.
#l1_missing = #ttcore.memory_space<l1>
func.func @composite_view_missing_logical_sizes() -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1_missing> {
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_missing>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_missing>
  // expected-error @+1 {{missing logicalSizes attr.}}
  %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32}> : (memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_missing>, memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_missing>) -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1_missing>
  return %0 : memref<4x1x32x32xf32, #ttcore.view<4>, #l1_missing>
}

// -----

// Rule 8 = logicalSizes length != number of inputs.
#l1_wronglen = #ttcore.memory_space<l1>
func.func @composite_view_wrong_logical_sizes_length() -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1_wronglen> {
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_wronglen>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_wronglen>
  // expected-error @+1 {{wrong logicalSizes length.}}
  %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32, logicalSizes = array<i64: 16, 16, 16>}> : (memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_wronglen>, memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_wronglen>) -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1_wronglen>
  return %0 : memref<4x1x32x32xf32, #ttcore.view<4>, #l1_wronglen>
}

// -----

// Rule 9 = sum of logicalSizes exceeds output capacity.
#l1_oversum = #ttcore.memory_space<l1>
func.func @composite_view_logical_sizes_exceed_capacity() -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1_oversum> {
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_oversum>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_oversum>
  // expected-error @+1 {{sum of logicalSizes exceeds output capacity.}}
  %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32, logicalSizes = array<i64: 40, 40>}> : (memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_oversum>, memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1_oversum>) -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1_oversum>
  return %0 : memref<4x1x32x32xf32, #ttcore.view<4>, #l1_oversum>
}

// -----

// Rule 10 = row-major width-concat per-DMA byte count not 16B-aligned.
#l1_align = #ttcore.memory_space<l1>
func.func @composite_view_row_major_width_alignment() -> memref<4x1x32x6xf32, #ttcore.view<4>, #l1_align> {
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<4x1x32x3xf32, #ttcore.shard<12x4, 1>, #l1_align>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<4x1x32x3xf32, #ttcore.shard<12x4, 1>, #l1_align>
  // expected-error @+1 {{row-major width-concat requires each input's per-DMA byte count}}
  %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32, logicalSizes = array<i64: 3, 3>}> : (memref<4x1x32x3xf32, #ttcore.shard<12x4, 1>, #l1_align>, memref<4x1x32x3xf32, #ttcore.shard<12x4, 1>, #l1_align>) -> memref<4x1x32x6xf32, #ttcore.view<4>, #l1_align>
  return %0 : memref<4x1x32x6xf32, #ttcore.view<4>, #l1_align>
}

// -----

// Rule 11 = inputs mix view and non-view operands.
#l1_mix = #ttcore.memory_space<l1>
#map_mix = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @composite_view_mixed_view_and_nonview() -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_mix> {
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_mix>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_mix>
  %view_0 = d2m.view_layout %alloc_0 remapping = #map_mix : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_mix> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_mix>
  // expected-error @+1 {{inputs must be uniformly views (ViewOpInterface) or uniformly non-views; input 1 disagrees with input 0}}
  %0 = "d2m.composite_view"(%view_0, %alloc_1) <{dim = 1 : si32}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_mix>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_mix>) -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_mix>
  return %0 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_mix>
}

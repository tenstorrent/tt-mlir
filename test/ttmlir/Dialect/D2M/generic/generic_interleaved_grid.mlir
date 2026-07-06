// RUN: ttmlir-opt %s --split-input-file --verify-diagnostics

// Validates that the d2m.generic grid-divisibility check is skipped for
// interleaved operands (which are not sharded across the op grid), but still
// enforced for sharded operands.

#dram = #ttcore.memory_space<dram>

// Interleaved output has a 1x1 grid while the op grid is 8x8 (the persistent /
// streaming form: every core streams tiles from one interleaved DRAM buffer).
// The divisibility invariant does not apply to interleaved operands, so this
// verifies successfully.
func.func @interleaved_output_grid_not_divisible_ok() {
  %in = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram>
  %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram>
  d2m.generic {block_factors = [], grid = #ttcore.grid<8x8>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<compute>]}
      ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.interleaved<4096x4096>, #dram>) {
  ^bb0:
  }
  return
}

// -----

#dram = #ttcore.memory_space<dram>

// Sharded output keeps the divisibility invariant: a 1x1 shard grid is not
// divisible by the 8x8 op grid, so this must still error.
func.func @sharded_output_grid_not_divisible_error() {
  %in = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
  %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
  // expected-error @+1 {{output grid shape must be divisible by the generic op's grid shape}}
  d2m.generic {block_factors = [], grid = #ttcore.grid<8x8>,
               indexing_maps = [], iterator_types = [],
               threads = [#d2m.thread<compute>]}
      ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>) {
  ^bb0:
  }
  return
}

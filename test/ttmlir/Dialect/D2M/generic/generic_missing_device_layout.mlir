// RUN: ttmlir-opt %s --split-input-file --verify-diagnostics

// This test validates device layout checking in the d2m.generic verifier.

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

func.func @missing_device_layout_memref() {
  // This memref only has memory space, no device layout - should error
  %bad = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
  %out = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

  // expected-error @+1 {{memref operand must have a device layout attribute (e.g., #ttcore.shard or #ttcore.interleaved), but got: 'memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>'}}
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [#map, #map],
               iterator_types = [#parallel, #parallel],
               threads = [#d2m.thread<compute>]}
      ins(%bad : memref<2x4x!ttcore.tile<32x32, f32>, #l1>)
      outs(%out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^compute0(%cb_in: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
  }
  return
}

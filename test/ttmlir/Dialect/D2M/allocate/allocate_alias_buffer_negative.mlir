// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Verify that the alias_buffer verifier rejects malformed ops.

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'd2m.alias_buffer' op element type mismatch
module {
  func.func @alias_buffer_element_type_mismatch() {
    %a = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %r = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%r : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^unified0:
      // Element type mismatch: input is tile<32x32, f32>, result is bf16.
      %alias = d2m.alias_buffer %a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1xbf16, #l1>
    }
    return
  }
}

// -----

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'd2m.alias_buffer' op result rank (4) must be less than input rank (4)
module {
  func.func @alias_buffer_rank_not_reduced() {
    %a = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %r = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%r : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^unified0:
      // Result rank equals input rank -- should be strictly less.
      %alias = d2m.alias_buffer %a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }
}

// -----

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'd2m.alias_buffer' op result shape 2, 2 does not match expected shard shape
module {
  func.func @alias_buffer_wrong_shard_shape() {
    %a = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %r = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%r : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^unified0:
      // Result shape 2x2 does not match expected shard shape 1x1.
      %alias = d2m.alias_buffer %a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }
}

// -----

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'd2m.alias_buffer' op result memory space does not match input memory space
module {
  func.func @alias_buffer_memspace_mismatch() {
    %a = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %r = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%r : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^unified0:
      // Result memory space (DRAM) does not match input memory space (L1).
      %alias = d2m.alias_buffer %a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #dram>
    }
    return
  }
}

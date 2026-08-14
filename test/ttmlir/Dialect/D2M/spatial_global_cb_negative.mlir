// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

#l1 = #ttcore.memory_space<l1>
!slot = memref<1x1x!ttcore.tile<32x32, f32>, #l1>

// CHECK: zip mapping requires equal sender and receiver core counts
func.func @zip_count_mismatch() {
  %gcb = d2m.create_global_cb {
    mapping = #d2m.global_cb_mapping<zip,
      sender = #ttcore.core_range<(0, 0), (0, 0)>,
      receiver = #ttcore.core_range<(0, 1), (0, 2)>>,
    num_slots = 2 : i64
  } : !d2m.global_cb<!slot>
  return
}

// -----

#l1 = #ttcore.memory_space<l1>
!slot = memref<1x1x!ttcore.tile<32x32, f32>, #l1>

// CHECK: row_fanout mapping requires sender width 1
func.func @row_fanout_width() {
  %gcb = d2m.create_global_cb {
    mapping = #d2m.global_cb_mapping<row_fanout,
      sender = #ttcore.core_range<(0, 0), (1, 1)>,
      receiver = #ttcore.core_range<(0, 2), (1, 3)>>,
    num_slots = 2 : i64
  } : !d2m.global_cb<!slot>
  return
}

// -----

#l1 = #ttcore.memory_space<l1>
!slot = memref<1x1x!ttcore.tile<32x32, f32>, #l1>

// CHECK: zip mapping sender and receiver core ranges must be disjoint
func.func @zip_overlap() {
  %gcb = d2m.create_global_cb {
    mapping = #d2m.global_cb_mapping<zip,
      sender = #ttcore.core_range<(0, 0), (1, 1)>,
      receiver = #ttcore.core_range<(1, 1), (2, 2)>>,
    num_slots = 2 : i64
  } : !d2m.global_cb<!slot>
  return
}

// -----

#l1 = #ttcore.memory_space<l1>
!slot = memref<1x1x!ttcore.tile<32x32, f32>, #l1>

// CHECK: num_slots must be >= 1
func.func @num_slots_zero() {
  %gcb = d2m.create_global_cb {
    mapping = #d2m.global_cb_mapping<zip,
      sender = #ttcore.core_range<(0, 0), (0, 0)>,
      receiver = #ttcore.core_range<(0, 1), (0, 1)>>,
    num_slots = 0 : i64
  } : !d2m.global_cb<!slot>
  return
}

// -----

#l1 = #ttcore.memory_space<l1>
!slot = memref<1x1x!ttcore.tile<32x32, f32>, #l1>

// CHECK: global_cb senders must be contained in one spatial region
func.func @mapping_outside_spatial(
    %out0: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
    %out1: memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
  %gcb = d2m.create_global_cb {
    mapping = #d2m.global_cb_mapping<zip,
      sender = #ttcore.core_range<(0, 0), (0, 0)>,
      receiver = #ttcore.core_range<(2, 2), (2, 2)>>,
    num_slots = 1 : i64
  } : !d2m.global_cb<!slot>

  d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>,
                              #ttcore.core_range<(0, 1), (0, 1)>]}
      ins()
      outs(%out0, %out1 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
                          memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
    ^region0:
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins()
          outs(%out0 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
          additionalArgs(%gcb : !d2m.global_cb<!slot>) {
        ^datamovement0:
          %src = memref.alloc() : !slot
          d2m.global_cb_reserve %gcb : !d2m.global_cb<!slot> -> !slot
          d2m.global_cb_push %gcb, %src : !d2m.global_cb<!slot>, !slot
      }, {
        ^compute0:
      }
  }, {
    ^region1:
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins()
          outs(%out1 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
          additionalArgs(%gcb : !d2m.global_cb<!slot>) {
        ^datamovement0:
          %got = d2m.global_cb_wait %gcb : !d2m.global_cb<!slot> -> !slot
          d2m.global_cb_pop %gcb : !d2m.global_cb<!slot>
      }, {
        ^compute0:
      }
  }
  return
}

// -----

#l1 = #ttcore.memory_space<l1>
!slot = memref<1x1x!ttcore.tile<32x32, f32>, #l1>

// CHECK: must be in a datamovement region
func.func @wrong_thread(
    %out0: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
    %out1: memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
  %gcb = d2m.create_global_cb {
    mapping = #d2m.global_cb_mapping<zip,
      sender = #ttcore.core_range<(0, 0), (0, 0)>,
      receiver = #ttcore.core_range<(0, 1), (0, 1)>>,
    num_slots = 1 : i64
  } : !d2m.global_cb<!slot>

  d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>,
                              #ttcore.core_range<(0, 1), (0, 1)>]}
      ins()
      outs(%out0, %out1 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
                          memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
    ^region0:
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins()
          outs(%out0 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
          additionalArgs(%gcb : !d2m.global_cb<!slot>) {
        ^datamovement0:
      }, {
        ^compute0:
          %src = memref.alloc() : !slot
          d2m.global_cb_reserve %gcb : !d2m.global_cb<!slot> -> !slot
          d2m.global_cb_push %gcb, %src : !d2m.global_cb<!slot>, !slot
      }
  }, {
    ^region1:
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins()
          outs(%out1 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
          additionalArgs(%gcb : !d2m.global_cb<!slot>) {
        ^datamovement0:
          %got = d2m.global_cb_wait %gcb : !d2m.global_cb<!slot> -> !slot
          d2m.global_cb_pop %gcb : !d2m.global_cb<!slot>
      }, {
        ^compute0:
      }
  }
  return
}

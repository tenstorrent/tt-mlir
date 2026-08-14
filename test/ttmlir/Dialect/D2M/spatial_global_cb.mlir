// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s

#l1 = #ttcore.memory_space<l1>
!slot = memref<1x1x!ttcore.tile<32x32, f32>, #l1>

// CHECK-LABEL: func.func @zip_8x4
func.func @zip_8x4(%out0: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
                   %out1: memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
  // CHECK: d2m.create_global_cb
  // CHECK-SAME: mapping = #d2m.global_cb_mapping<zip, sender = #ttcore.core_range<(0,0), (7,3)>, receiver = #ttcore.core_range<(0,4), (7,7)>>
  // CHECK-SAME: num_slots = 2
  %gcb = d2m.create_global_cb {
    mapping = #d2m.global_cb_mapping<zip,
      sender = #ttcore.core_range<(0, 0), (7, 3)>,
      receiver = #ttcore.core_range<(0, 4), (7, 7)>>,
    num_slots = 2 : i64
  } : !d2m.global_cb<!slot>

  d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (7, 3)>,
                              #ttcore.core_range<(0, 4), (7, 7)>]}
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

// CHECK-LABEL: func.func @row_fanout
func.func @row_fanout(%out0: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
                      %out1: memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
  // CHECK: mapping = #d2m.global_cb_mapping<row_fanout, sender = #ttcore.core_range<(0,0), (1,0)>, receiver = #ttcore.core_range<(0,1), (1,2)>>
  %gcb = d2m.create_global_cb {
    mapping = #d2m.global_cb_mapping<row_fanout,
      sender = #ttcore.core_range<(0, 0), (1, 0)>,
      receiver = #ttcore.core_range<(0, 1), (1, 2)>>,
    num_slots = 2 : i64
  } : !d2m.global_cb<!slot>

  d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (1, 0)>,
                              #ttcore.core_range<(0, 1), (1, 2)>]}
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

// CHECK-LABEL: func.func @explicit_pairs
func.func @explicit_pairs(%out0: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
                          %out1: memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
  // CHECK: mapping = #d2m.global_cb_mapping<explicit, [#d2m.sender_receivers<(0,0), [<(0,1), (0,1)>]>]>
  %gcb = d2m.create_global_cb {
    mapping = #d2m.global_cb_mapping<explicit, [
      #d2m.sender_receivers<(0, 0), [#ttcore.core_range<(0, 1), (0, 1)>]>
    ]>,
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

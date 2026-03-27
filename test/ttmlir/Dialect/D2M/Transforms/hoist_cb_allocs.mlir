// RUN: ttmlir-opt --d2m-hoist-cb-allocs -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that d2m-hoist-cb-allocs creates deallocs after the generic for
// each hoisted CB alloc.

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#remap4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @hoist_cb_dealloc(
    %stream: memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>,
    %out: memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  // CHECK-LABEL: func @hoist_cb_dealloc
  // CHECK: %[[CB:.*]] = memref.alloc() {address = 1024 : i64{{.*}}cb_layout
  // CHECK: d2m.generic
  // CHECK: additionalArgs(%[[CB]]
  // CHECK: memref.dealloc %[[CB]]
  d2m.generic {block_factors = [], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
      ins(%stream : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
      outs(%out : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^unified0():
    %c0 = d2m.core_index(0) : index
    %c1 = d2m.core_index(1) : index
    %alloc_cb = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %0 = d2m.remote_load %alloc_cb %stream[%c0, %c1] : memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> -> memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %1 = d2m.remote_store %out[%c0, %c1] %0 : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1> -> memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  }
  return
}

// Verify that multiple hoisted allocs each get their own dealloc.
func.func @hoist_multiple_cbs_dealloc(
    %stream0: memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>,
    %stream1: memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>,
    %out: memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  // CHECK-LABEL: func @hoist_multiple_cbs_dealloc
  // CHECK: %[[CB0:.*]] = memref.alloc() {address = 1024 : i64{{.*}}cb_layout
  // CHECK: %[[CB1:.*]] = memref.alloc() {address = 2048 : i64{{.*}}cb_layout
  // CHECK: d2m.generic
  // CHECK: additionalArgs(%[[CB0]], %[[CB1]]
  // CHECK: memref.dealloc %[[CB1]]
  // CHECK: memref.dealloc %[[CB0]]
  d2m.generic {block_factors = [], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
      ins(%stream0, %stream1 : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
      outs(%out : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^unified0():
    %c0 = d2m.core_index(0) : index
    %c1 = d2m.core_index(1) : index
    %alloc_cb0 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %0 = d2m.remote_load %alloc_cb0 %stream0[%c0, %c1] : memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> -> memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_cb1 = memref.alloc() {address = 2048 : i64, alignment = 16 : i64} : memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %1 = d2m.remote_load %alloc_cb1 %stream1[%c0, %c1] : memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> -> memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %2 = d2m.remote_store %out[%c0, %c1] %0 : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1> -> memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  }
  return
}

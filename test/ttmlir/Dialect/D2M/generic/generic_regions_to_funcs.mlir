// RUN: ttmlir-opt --ttcore-register-device --cse --d2m-normalize-thread-args --d2m-generic-regions-to-funcs -o %t %s
// RUN: FileCheck %s --input-file=%t --check-prefix=CHECK
// RUN: ttmlir-opt --ttcore-register-device --d2m-normalize-thread-args --d2m-generic-regions-to-funcs -o %t.maps %s
// RUN: FileCheck %s --input-file=%t.maps --check-prefix=MAPS

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#remap = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>
#grid_v2p = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>

ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

func.func @add(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1_>, %arg1: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1_> {
  %outer_c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1_>

  %view = d2m.view_layout %arg0 remapping = #remap :
       memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #ttcore.memory_space<l1>>
    -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>

  // CHECK: [#d2m.thread<datamovement, @datamovement_kernel0, dm_core = 2>, #d2m.thread<compute, @compute_kernel1>]
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, dm_core = 2>, #d2m.thread<compute>]}
               ins(%view, %arg1 :
                memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>,
                memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1_>)
               outs(%alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1_>) {
  ^datamovement0:
    %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
    %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
    %cb2 = d2m.get_cb(2) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %c0 = arith.constant 0 : index
    %tx = d2m.dma_read %view[%c0, %c0, %c0], %mem1[%c0], <1> : (
      memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>,
      memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx<read>
    d2m.dma_wait %tx : !d2m.mem_tx<read>
  }, {
  ^compute0:
  %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
  %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
  %cb2 = d2m.get_cb(2) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
  }
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1_>
}
// CHECK: func.func private @datamovement_kernel0{{.*}} attributes {d2m.thread = #d2m.thread<datamovement, dm_core = 2>, tt.function_type = "kernel"}
// CHECK: d2m.get_arg(0)
// CHECK: arith.constant 0 : index
// CHECK: func.func private @compute_kernel1{{.*}} attributes {d2m.thread = #d2m.thread<compute>, tt.function_type = "kernel"}

func.func @core_index_maps_during_outlining(%arg0: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>
  d2m.generic {block_factors = [], grid = #grid_v2p, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute>]}
      ins(%arg0 : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
      outs(%alloc : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
  ^compute0:
    %core0 = d2m.core_index(0) : index
    %core1 = d2m.core_index(1) : index
  }
  return
}

// MAPS: func.func private @compute_kernel2{{.*}} attributes {d2m.thread = #d2m.thread<compute>, tt.function_type = "kernel"}
// MAPS: d2m.core_index(0) {phys_to_virt_map = {{.*}}} : index
// MAPS: d2m.core_index(1) {phys_to_virt_map = {{.*}}} : index

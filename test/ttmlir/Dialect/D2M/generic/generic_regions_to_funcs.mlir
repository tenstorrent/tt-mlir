// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-regions-to-funcs -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

func.func @add(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0>, #l1_>, %arg1: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0>, #l1_>

  %view = d2m.view_layout %arg0 :
       memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0>, #ttcore.memory_space<l1>>
    -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, #ttcore.memory_space<l1>>

  // CHECK: [#d2m.thread<datamovement, @datamovement_kernel0>, #d2m.thread<compute, @compute_kernel1>]
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
               ins(%view, %arg1 :
                memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, #ttcore.memory_space<l1>>,
                memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0>, #l1_>)
               outs(%alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0>, #l1_>) {
  ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %c0 = arith.constant 0 : index
    %tx = d2m.dma_read %view[%c0, %c0, %c0], %mem1[%c0], <1> : (
      memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, #ttcore.memory_space<l1>>,
      memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
  }
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0>, #l1_>
}
// CHECK: func.func private @datamovement_kernel0{{.*}} attributes {d2m.thread = #d2m.thread<datamovement>}
// CHECK: d2m.get_global_operand(0)
// CHECK: func.func private @compute_kernel1{{.*}} attributes {d2m.thread = #d2m.thread<compute>}

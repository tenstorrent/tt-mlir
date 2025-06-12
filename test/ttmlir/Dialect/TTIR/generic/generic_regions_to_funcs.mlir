// RUN: ttmlir-opt --tt-register-device --ttir-generic-regions-to-funcs %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
#dram = #tt.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>

func.func @add(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<0x0>, #l1_>, %arg1: memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<0x0>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<0x0>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<0x0>, #l1_>
  // CHECK: [#ttir.thread<datamovement, @datamovement_kernel0>, #ttir.thread<compute, @compute_kernel1>]
  ttir.generic {block_factors = [1, 1], grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<datamovement>, #ttir.thread<compute>]}
               ins(%arg0, %arg1 : memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<0x0>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<0x0>, #l1_>)
               outs(%alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<0x0>, #l1_>) {
  ^datamovement0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    %c0 = arith.constant 0 : index
    %tx = ttir.dma %arg0[%c0, %c0], %cb0 : (memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<0x0>, #l1_>, memref<2x4x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
    "ttir.yield"() : () -> ()
  }, {
  ^compute0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    "ttir.yield"() : () -> ()
  }
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<0x0>, #l1_>
}
// CHECK: func.func private @datamovement_kernel0{{.*}} attributes {ttir.thread = #ttir.thread<datamovement>}
// CHECK: ttir.get_global_operand(0)
// CHECK: func.func private @compute_kernel1{{.*}} attributes {ttir.thread = #ttir.thread<compute>}

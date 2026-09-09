// RUN: not ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access-scheduled %s 2>&1 | FileCheck %s

#l1_ = #ttcore.memory_space<l1>

module {
  func.func @dynamic_affine_bound_rejected(
      %ub: index,
      %in0: memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %init: memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %out0: memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %init : memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>>
      %cb2_raw = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      %cb1 = d2m.wait %cb1_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      %cb2 = d2m.reserve %cb2_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      affine.for %i = 0 to %ub {
        %a = affine.load %cb0[0, 0] : memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
        %c = affine.load %cb1[0, 0] : memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
        // CHECK: error: 'd2m.tile_sfpu_reduce_sum' op requires constant-bounded affine.for loops for DST register access linearization
        %r = "d2m.tile_sfpu_reduce_sum"(%a, %c) {reduce_dim = #d2m<reduce_dim R>} : (!ttcore.tile<32x32, si32>, !ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
        affine.store %r, %cb2[0, 0] : memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      }
    }
    return
  }
}

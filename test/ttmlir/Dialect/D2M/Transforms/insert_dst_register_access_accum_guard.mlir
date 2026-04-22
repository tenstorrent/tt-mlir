// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access-unscheduled --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @unary_outer_loop_is_not_accum
  // CHECK: affine.for %[[BLOCK:.*]] = 0 to 2 {
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #dst>
  // CHECK-NOT: scf.if
  // CHECK: %[[DST_VAL:.*]] = affine.load %[[DST]][0] : memref<4x!ttcore.tile<32x32, f32>, #dst>
  // CHECK: %[[NEG:.*]] = "d2m.tile_negative"(%[[DST_VAL]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
  func.func @unary_outer_loop_is_not_accum(
      %in: memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %out: memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [2, 1], grid = #ttcore.grid<1x1>,
                 indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                  affine_map<(d0, d1) -> (d0, d1)>],
                 iterator_types = [#ttcore.iterator_type<parallel>,
                                   #ttcore.iterator_type<parallel>],
                 threads = [#d2m.thread<unified>]}
        ins(%in : memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^unified0:
      %cb0 = d2m.get_cb(0) operand_index = 0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %cb1 = d2m.get_cb(1) operand_index = 1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %in_cb = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %out_cb = d2m.reserve %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      affine.for %blocked = 0 to 2 {
        affine.for %inner = 0 to 1 {
          %tile = affine.load %in_cb[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
          %neg = "d2m.tile_negative"(%tile) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %neg, %out_cb[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
        } {d2m.linalg_root}
      }
    }
    func.return
  }
}

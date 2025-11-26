// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc --split-input-file %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=chaitin-briggs" --split-input-file %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=greedy" --split-input-file %s | FileCheck %s
//
// Tests for in-place unary operations with graph coloring DST allocation.
// Unary ops like tile_exp have getDstRegInPlace() == true, meaning the input
// and output must use the same DST register index. This tests that the graph
// coloring pass correctly coalesces these constraints.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // Test: Single unary operation (tile_exp) with in-place semantics.
  // The input store and final output load must use the same DST index.
  //
  // CHECK-LABEL: func.func @unary_exp_single
  // CHECK: d2m.generic
  // CHECK: %[[MEM0:.*]] = d2m.wait
  // CHECK: %[[MEM_OUT:.*]] = d2m.reserve
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst>
  // CHECK: affine.load %[[MEM0]]
  // CHECK: affine.store {{.*}}, %[[DST]][[[REG_IDX_0:[0-9]+]],
  // CHECK: affine.load %[[DST]]
  // CHECK: "d2m.tile_exp"{{.*}}{result_dst_index = [[REG_IDX_0]] : i64}
  // CHECK: affine.store {{.*}}, %[[DST]]
  // CHECK: affine.load %[[DST]][[[REG_IDX_0]],
  // CHECK: affine.store {{.*}}, %[[MEM_OUT]]
  // CHECK: d2m.release_dst %[[DST]]

  func.func @unary_exp_single(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                              %out: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %c0 = arith.constant 0 : index
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

      %v0 = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %result = "d2m.tile_exp"(%v0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      affine.store %result, %mem_out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}

// -----

// Tests chain of in-place unary operations.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // Test: Chain of unary operations (abs -> neg -> exp).
  // In-place unary ops are chained without intermediate DST stores.
  // The final result uses the same DST index as the input (coalescing).
  //
  // CHECK-LABEL: func.func @unary_chain
  // CHECK: d2m.generic
  // CHECK: %[[MEM0:.*]] = d2m.wait
  // CHECK: %[[MEM_OUT:.*]] = d2m.reserve
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst>
  // CHECK: affine.load %[[MEM0]]
  // CHECK: affine.store {{.*}}, %[[DST]][[[REG_IDX_0:[0-9]+]],
  // CHECK: affine.load %[[DST]]
  // CHECK: "d2m.tile_abs"
  // CHECK: "d2m.tile_negative"
  // CHECK: "d2m.tile_exp"{{.*}}{result_dst_index = [[REG_IDX_0]] : i64}
  // CHECK: affine.store {{.*}}, %[[DST]]
  // CHECK: affine.load %[[DST]][[[REG_IDX_0]],
  // CHECK: affine.store {{.*}}, %[[MEM_OUT]]
  // CHECK: d2m.release_dst %[[DST]]

  func.func @unary_chain(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                         %out: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %c0 = arith.constant 0 : index
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

      %v0 = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %abs = "d2m.tile_abs"(%v0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %neg = "d2m.tile_negative"(%abs) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %exp = "d2m.tile_exp"(%neg) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      affine.store %exp, %mem_out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}

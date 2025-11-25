// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=greedy" --split-input-file %s | FileCheck %s --check-prefixes=COMMON,GREEDY
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=chaitin-briggs" --split-input-file %s | FileCheck %s --check-prefixes=COMMON,CHAITIN
//
// Tests for graph coloring DST allocation on unmarked loops/scalar access.
// These tests use processUnmarkedRegion and are GC-only (no legacy pass support).

#l1_ = #ttcore.memory_space<l1>

module {
  // COMMON-LABEL: func.func @test_binary_op_single_tile
  // COMMON: d2m.generic
  // COMMON: ^compute0(%[[CB0:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>, %[[CB1:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>, %[[CB_OUT:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>):

  // COMMON: %[[MEM0:.*]] = d2m.wait %[[CB0]]
  // COMMON: %[[MEM1:.*]] = d2m.wait %[[CB1]]
  // COMMON: %[[MEM_OUT:.*]] = d2m.reserve %[[CB_OUT]]

  // Graph coloring allocates slices for v0, v1, and result (3 slices total)
  // COMMON: %[[DST:.*]] = d2m.acquire_dst() : memref<3x1x1x!ttcore.tile<32x32, f16>, #dst>

  // Load from L1 input 0, store to DST slice 0
  // COMMON: affine.load %[[MEM0]]
  // COMMON: affine.store {{.*}}, %[[DST]][0,

  // Load from DST
  // COMMON: affine.load %[[DST]]

  // Load from L1 input 1, store to DST slice 1
  // COMMON: affine.load %[[MEM1]]
  // COMMON: affine.store {{.*}}, %[[DST]][1,

  // Load from DST
  // COMMON: affine.load %[[DST]]

  // Compute operation uses values loaded from DST
  // COMMON: "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}

  // Store result to DST slice 2, load from DST, then store to L1 output
  // COMMON: affine.store {{.*}}, %[[DST]]
  // COMMON: affine.load %[[DST]][2,
  // COMMON: affine.store {{.*}}, %[[MEM_OUT]]

  // COMMON: d2m.release_dst %[[DST]]

  func.func @test_binary_op_single_tile(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
                                %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
                                %out: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1 :
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
      %c0 = arith.constant 0 : index

      %0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %out_mem = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>

      %v0 = affine.load %0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %v1 = affine.load %1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %result = "d2m.tile_add"(%v0, %v1) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>

      affine.store %result, %out_mem[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
    }
    return
  }
}

// -----

// Tests unary chain with intermediate elision showing strategy differences.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // COMMON-LABEL: func.func @test_unary_chain_intermediate_elision
  // COMMON: d2m.generic
  // COMMON: %[[MEM0:.*]] = d2m.wait
  // COMMON: %[[MEM1:.*]] = d2m.wait
  // COMMON: %[[MEM_OUT:.*]] = d2m.reserve
  // COMMON: %[[DST:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst>
  // COMMON: affine.load %[[MEM0]]
  // COMMON: affine.load %[[MEM1]]
  // COMMON: "d2m.tile_sub"
  // COMMON: "d2m.tile_eqz"
  // CHAITIN: "d2m.tile_eqz"{{.*}}{result_dst_index = 1 : i64}
  // GREEDY: "d2m.tile_eqz"{{.*}}{result_dst_index = 0 : i64}
  // COMMON: affine.store {{.*}}, %[[MEM_OUT]]
  // COMMON: d2m.release_dst %[[DST]] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst>

  func.func @test_unary_chain_intermediate_elision(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                        %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                        %out: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1 :
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %c0 = arith.constant 0 : index
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

      %v0 = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %v1 = affine.load %mem1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %sub_result = "d2m.tile_sub"(%v0, %v1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %eqz1_result = "d2m.tile_eqz"(%sub_result) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %eqz2_result = "d2m.tile_eqz"(%eqz1_result) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      affine.store %eqz2_result, %mem_out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}

// -----

// Tests three-input strategy differences showing greedy vs chaitin-briggs allocation.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // COMMON-LABEL: func.func @test_three_input_strategy_differences
  // COMMON: d2m.generic
  // COMMON: %[[MEM0:.*]] = d2m.wait
  // COMMON: %[[MEM1:.*]] = d2m.wait
  // COMMON: %[[MEM2:.*]] = d2m.wait
  // COMMON: %[[MEM_OUT:.*]] = d2m.reserve
  // COMMON: %[[DST:.*]] = d2m.acquire_dst() : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst>

  // Copy inputs to DST and load back
  // COMMON: affine.load %[[MEM0]]
  // COMMON: affine.store {{.*}}, %[[DST]][0,
  // COMMON: affine.load %[[DST]]
  // COMMON: affine.load %[[MEM1]]
  // COMMON: affine.store {{.*}}, %[[DST]][1,
  // COMMON: affine.load %[[DST]]
  // COMMON: affine.load %[[MEM2]]
  // COMMON: affine.store {{.*}}, %[[DST]][2,
  // COMMON: affine.load %[[DST]]

  // Compute operations
  // COMMON: "d2m.tile_abs"
  // COMMON: "d2m.tile_sin"
  // COMMON: "d2m.tile_negative"
  // COMMON: "d2m.tile_add"
  // GREEDY: "d2m.tile_mul"{{.*}}{result_dst_index = 0 : i64}
  // CHAITIN: "d2m.tile_mul"{{.*}}{result_dst_index = 2 : i64}

  // Store result and copy to L1
  // COMMON: affine.store {{.*}}, %[[DST]]
  // GREEDY: affine.load %[[DST]][0,
  // CHAITIN: affine.load %[[DST]][2,
  // COMMON: affine.store {{.*}}, %[[MEM_OUT]]

  // COMMON: d2m.release_dst %[[DST]]

  func.func @test_three_input_strategy_differences(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                              %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                              %in2: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                              %out: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1, %in2 :
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %c0 = arith.constant 0 : index
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem2 = d2m.wait %cb2 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

      %v0 = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %v1 = affine.load %mem1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %v2 = affine.load %mem2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

      // Three independent operations
      %abs_v0 = "d2m.tile_abs"(%v0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %sin_v1 = "d2m.tile_sin"(%v1) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %neg_v2 = "d2m.tile_negative"(%v2) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>

      // Binary op combining results of second two independent ops
      %add_abs_sin = "d2m.tile_add"(%neg_v2, %sin_v1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>

      // Final op combining previous result with first independent result
      %final = "d2m.tile_mul"(%add_abs_sin, %abs_v0) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>

      affine.store %final, %mem_out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}

// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc --split-input-file %s | FileCheck %s
//
// Verifies that the d2m-insert-dst-register-gc pass does the following:
//   1. acquire_dst is created
//   2. L1→DST copy loops are generated before operations
//   3. Original loads are replaced with DST loads
//   4. DST→L1 copy loops are generated for results
//   5. Graph coloring assigns optimal slice indices avoiding overwrites (v0→slice 1, v1→slice 0, result→slice 0)

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @test_linalg_input
  // CHECK: d2m.generic
  // CHECK: ^compute0(%[[CB0:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>, %[[CB1:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>, %[[CB_OUT:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>):

  // Graph coloring optimizes slice allocation - result can reuse a slice from an input
  // that's no longer live, so only 2 slices are needed instead of 3.
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index

  // CHECK: %[[MEM0:.*]] = d2m.wait %[[CB0]]
  // CHECK-NEXT: %[[MEM1:.*]] = d2m.wait %[[CB1]]
  // CHECK-NEXT: %[[MEM_OUT:.*]] = d2m.reserve %[[CB_OUT]]

  // Load from L1 input 0, store to DST slice 1, then load from DST
  // Graph coloring assigns slice 1 to v0
  // CHECK: %[[L1_VAL0:.*]] = affine.load %[[MEM0]][%[[C0]], %[[C0]]]
  // CHECK-NEXT: affine.store %[[L1_VAL0]], %[[DST]][1, %[[C0]], %[[C0]]]
  // CHECK-NEXT: %[[DST_VAL0:.*]] = affine.load %[[DST]][1, %[[C0]], %[[C0]]]

  // Load from L1 input 1, store to DST slice 0, then load from DST
  // Graph coloring assigns slice 0 to v1 (different from v0)
  // CHECK: %[[L1_VAL1:.*]] = affine.load %[[MEM1]][%[[C0]], %[[C0]]]
  // CHECK-NEXT: affine.store %[[L1_VAL1]], %[[DST]][0, %[[C0]], %[[C0]]]
  // CHECK-NEXT: %[[DST_VAL1:.*]] = affine.load %[[DST]][0, %[[C0]], %[[C0]]]

  // Compute operation uses values loaded from DST
  // CHECK: %[[RESULT:.*]] = "d2m.tile_add"(%[[DST_VAL0]], %[[DST_VAL1]])

  // Store result to DST slice 0 (reuses v1's slice), load from DST, then store to L1 output
  // Graph coloring reuses slice 0 for the result since v1 is no longer live after being loaded
  // CHECK-NEXT: affine.store %[[RESULT]], %[[DST]][0, %[[C0]], %[[C0]]]
  // CHECK-NEXT: %[[DST_RESULT:.*]] = affine.load %[[DST]][0, %[[C0]], %[[C0]]]
  // CHECK-NEXT: affine.store %[[DST_RESULT]], %[[MEM_OUT]][%[[C0]], %[[C0]]]

  // CHECK: d2m.release_dst %[[DST]]

  func.func @test_linalg_input(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
                                %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
                                %out: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
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
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
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

// Verifies that linalg.generic operations are NOT transformed by this pass.
// Linalg operations should be converted to affine loops by a separate pass.

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @test_with_linalg
  // CHECK: d2m.generic
  // CHECK: linalg.generic
  // CHECK-NOT: d2m.acquire_dst
  // CHECK-NOT: affine.store

  func.func @test_with_linalg(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
                               %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
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
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %c0 = arith.constant 0 : index
      %subview0 = memref.subview %mem0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
      %subview1 = memref.subview %mem1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
      %subview_out = memref.subview %mem_out[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
      linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%subview0, %subview1 : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>,
                                   memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>)
        outs(%subview_out : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f16>, %arg1: !ttcore.tile<32x32, f16>, %arg2: !ttcore.tile<32x32, f16>):
        %result = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
        linalg.yield %result : !ttcore.tile<32x32, f16>
      }
    }
    return
  }
}

// -----

// Verifies that DST allocation works correctly with nested affine loops.
// Operations inside loops should be detected as interfering with each other.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @test_with_affine_loops
  // CHECK: d2m.generic
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: affine.for
  // CHECK: affine.for
  // CHECK: affine.store {{.*}}, %[[DST]][1,
  // CHECK: affine.store {{.*}}, %[[DST]][0,
  // CHECK: affine.store {{.*}}, %[[DST]][0,
  // CHECK: d2m.release_dst

  func.func @test_with_affine_loops(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
                                     %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
                                     %out: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
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
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>

      affine.for %i = 0 to 1 {
        affine.for %j = 0 to 1 {
          %v0 = affine.load %mem0[%i, %j] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v1 = affine.load %mem1[%i, %j] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %result = "d2m.tile_add"(%v0, %v1) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          affine.store %result, %mem_out[%i, %j] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
        }
      }
    }
    return
  }
}

// -----

// Verifies f32 data type support (similar to @binary from insert_dst_register_access_eltwise_large_dst.mlir).

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @test_f32_binary
  // CHECK: d2m.generic
  // Graph coloring allocates 2 slices: one for each input (result reuses slice 0)
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst>
  // CHECK: affine.load
  // CHECK: affine.store {{.*}}, %[[DST]]
  // CHECK: "d2m.tile_maximum"
  // CHECK: d2m.release_dst %[[DST]] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst>

  func.func @test_f32_binary(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                              %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                              %out: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
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
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %c0 = arith.constant 0 : index
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

      %v0 = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %v1 = affine.load %mem1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %result = "d2m.tile_maximum"(%v0, %v1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      affine.store %result, %mem_out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}

// -----

// Verifies operation chains with intermediate results (similar to @intermediates_thru_dst_chain_2).
// Tests that intermediate values are properly allocated and reused through graph coloring.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @test_operation_chain
  // CHECK: d2m.generic
  // Graph coloring allocates 2 slices: inputs are loaded and consumed, intermediate result
  // (tile_div output) is used directly by tile_recip without needing DST storage
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst>
  // CHECK: "d2m.tile_div"
  // CHECK: "d2m.tile_recip"
  // CHECK: affine.store {{.*}}, %[[DST]]
  // CHECK: d2m.release_dst %[[DST]] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst>

  func.func @test_operation_chain(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                                   %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                                   %out: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
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
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %c0 = arith.constant 0 : index
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

      %v0 = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %v1 = affine.load %mem1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %div_result = "d2m.tile_div"(%v0, %v1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %recip_result = "d2m.tile_recip"(%div_result) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      affine.store %recip_result, %mem_out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}

// -----

// Verifies longer operation chains with multiple intermediate values (similar to @intermediates_thru_dst_chain_3).

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @test_long_operation_chain
  // CHECK: d2m.generic
  // Graph coloring allocates 2 slices: inputs consumed early, intermediate results
  // (tile_sub, tile_eqz outputs) used directly without DST storage
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst>
  // CHECK: "d2m.tile_sub"
  // CHECK: "d2m.tile_eqz"
  // CHECK: "d2m.tile_eqz"
  // CHECK: d2m.release_dst %[[DST]] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst>

  func.func @test_long_operation_chain(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                                        %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                                        %out: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
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
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
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

// Verifies multi-tile unary operation chains (similar to @eltwise_unary_chain_multi_tile).
// Tests 2x2 tile processing with operation chains.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @test_unary_chain_multi_tile
  // CHECK: d2m.generic
  // Graph coloring with conservative loop interference: all operations in same loop interfere
  // Allocates 1 slice of 2x2 tiles (load and store reuse the same slice)
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst>
  // CHECK: affine.for
  // CHECK: affine.for
  // CHECK: "d2m.tile_abs"
  // CHECK: "d2m.tile_sin"
  // CHECK: "d2m.tile_negative"
  // CHECK: "d2m.tile_exp"
  // CHECK: d2m.release_dst %[[DST]] : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst>

  func.func @test_unary_chain_multi_tile(%in0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>,
                                          %out: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>)
      outs(%out : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb_out: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          %v0 = affine.load %mem0[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %abs_result = "d2m.tile_abs"(%v0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          %sin_result = "d2m.tile_sin"(%abs_result) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          %neg_result = "d2m.tile_negative"(%sin_result) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          %exp_result = "d2m.tile_exp"(%neg_result) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %exp_result, %mem_out[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
    }
    return
  }
}

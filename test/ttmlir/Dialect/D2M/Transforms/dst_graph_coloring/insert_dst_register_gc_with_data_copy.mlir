// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=greedy" --split-input-file %s | FileCheck %s --check-prefixes=COMMON,GREEDY
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=chaitin-briggs" --split-input-file %s | FileCheck %s --check-prefixes=COMMON,CHAITIN
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access --split-input-file %s | FileCheck %s --check-prefixes=COMMON,BASIC
//
// Verifies that the d2m-insert-dst-register-gc pass does the following:
//   1. acquire_dst is created
//   2. L1->DST copy loops are generated before operations
//   3. Original loads are replaced with DST loads
//   4. DST->L1 copy loops are generated for results
//   5. Graph coloring assigns optimal slice indices avoiding overwrites (v0->slice 1, v1->slice 0, result->slice 0)

#l1_ = #ttcore.memory_space<l1>

module {
  // COMMON-LABEL: func.func @test_no_loops
  // COMMON: d2m.generic
  // COMMON: ^compute0(%[[CB0:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>, %[[CB1:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>, %[[CB_OUT:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>):

  // Graph coloring allocates slices for v0, v1, and result (3 slices total)
  // COMMON: %[[DST:.*]] = d2m.acquire_dst() : memref<3x1x1x!ttcore.tile<32x32, f16>, #dst>
  // COMMON-NEXT: %[[C0:.*]] = arith.constant 0 : index

  // COMMON: %[[MEM0:.*]] = d2m.wait %[[CB0]]
  // COMMON-NEXT: %[[MEM1:.*]] = d2m.wait %[[CB1]]
  // COMMON-NEXT: %[[MEM_OUT:.*]] = d2m.reserve %[[CB_OUT]]

  // Load from L1 input 0, store to DST slice 0, then load from DST
  // COMMON: %[[L1_VAL0:.*]] = affine.load %[[MEM0]][%[[C0]], %[[C0]]]
  // COMMON-NEXT: affine.store %[[L1_VAL0]], %[[DST]][0, %[[C0]], %[[C0]]]
  // COMMON-NEXT: %[[DST_VAL0:.*]] = affine.load %[[DST]][0, %[[C0]], %[[C0]]]

  // Load from L1 input 1, store to DST slice 1, then load from DST
  // COMMON: %[[L1_VAL1:.*]] = affine.load %[[MEM1]][%[[C0]], %[[C0]]]
  // COMMON-NEXT: affine.store %[[L1_VAL1]], %[[DST]][1, %[[C0]], %[[C0]]]
  // COMMON-NEXT: %[[DST_VAL1:.*]] = affine.load %[[DST]][1, %[[C0]], %[[C0]]]

    // Compute operation uses values loaded from DST
  // Verify that result_dst_index attribute is attached for all strategies
  // BASIC: %[[RESULT:.*]] = "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}
  // CHAITIN: %[[RESULT:.*]] = "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}
  // GREEDY: %[[RESULT:.*]] = "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}

  // Store result to DST slice 2, load from DST, then store to L1 output
  // COMMON-NEXT: affine.store %[[RESULT]], %[[DST]][2, %[[C0]], %[[C0]]]
  // COMMON-NEXT: %[[DST_RESULT:.*]] = affine.load %[[DST]][2, %[[C0]], %[[C0]]]
  // COMMON-NEXT: affine.store %[[DST_RESULT]], %[[MEM_OUT]][%[[C0]], %[[C0]]]

  // COMMON: d2m.release_dst %[[DST]]

  func.func @test_no_loops(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
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

// Verifies that DST allocation works correctly with nested affine loops.
// Operations inside loops should be detected as interfering with each other.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // COMMON-LABEL: func.func @test_with_affine_loops
  // COMMON: d2m.generic
  // COMMON: ^compute0(%[[CB0:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>, %[[CB1:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>, %[[CB_OUT:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>):
  // COMMON: %[[DST:.*]] = d2m.acquire_dst() : memref<3x1x1x!ttcore.tile<32x32, f16>, #dst>
  // COMMON: %[[MEM0:.*]] = d2m.wait %[[CB0]]
  // COMMON: %[[MEM1:.*]] = d2m.wait %[[CB1]]
  // COMMON: %[[MEM_OUT:.*]] = d2m.reserve %[[CB_OUT]]

  // Verify outer loop with three inner loops inside
  // COMMON: affine.for %[[I:.*]] = 0 to 1 {

  // PROLOGUE: L1->DST copies
  // COMMON: affine.for %[[J_PROLOGUE:.*]] = 0 to 1 {
  // COMMON: %[[L1_VAL0:.*]] = affine.load %[[MEM0]][%[[I]], %[[J_PROLOGUE]]]
  // COMMON: affine.store %[[L1_VAL0]], %[[DST]][0, %[[I]], %[[J_PROLOGUE]]]
  // COMMON: %[[L1_VAL1:.*]] = affine.load %[[MEM1]][%[[I]], %[[J_PROLOGUE]]]
  // COMMON: affine.store %[[L1_VAL1]], %[[DST]][1, %[[I]], %[[J_PROLOGUE]]]
  // COMMON: }

  // COMPUTE: DST operations
  // COMMON: affine.for %[[J_COMPUTE:.*]] = 0 to 1 {
  // COMMON: affine.load %[[DST]][0, %[[I]], %[[J_COMPUTE]]]]
  // COMMON: affine.load %[[DST]][1, %[[I]], %[[J_COMPUTE]]]]
  // BASIC: %[[RESULT:.*]] = "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}
  // CHAITIN: %[[RESULT:.*]] = "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}
  // GREEDY: %[[RESULT:.*]] = "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}
  // COMMON: affine.store %[[RESULT]], %[[DST]][2, %[[I]], %[[J_COMPUTE]]]
  // COMMON: }

  // EPILOGUE: DST->L1 copies
  // COMMON: affine.for %[[J_EPILOGUE:.*]] = 0 to 1 {
  // COMMON: %[[DST_RESULT:.*]] = affine.load %[[DST]][2, %[[I]], %[[J_EPILOGUE]]]
  // COMMON: affine.store %[[DST_RESULT]], %[[MEM_OUT]][%[[I]], %[[J_EPILOGUE]]]
  // COMMON: }

  // COMMON: }
  // COMMON: d2m.release_dst %[[DST]]

  func.func @test_with_affine_loops(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
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

// Verifies f32 data type support with d2m.generic wrapper and result_dst_index checks.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // COMMON-LABEL: func.func @test_f32_binary
  // COMMON: d2m.generic
  // COMMON: %[[DST:.*]] = d2m.acquire_dst
  // COMMON: affine.for
  // COMMON: affine.load {{.*}}: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  // COMMON: affine.store {{.*}}: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  // BASIC: "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}
  // CHAITIN: "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}
  // GREEDY: "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}
  // COMMON: d2m.release_dst %[[DST]]

  func.func @test_f32_binary(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
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

      affine.for %i = 0 to 1 {
        affine.for %j = 0 to 1 {
          %0 = affine.load %mem0[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
          %1 = affine.load %mem1[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
          %2 = "d2m.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %2, %mem_out[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
    }
    return
  }
}

// -----

// Verifies longer operation chains with multiple intermediate values.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // COMMON-LABEL: func.func @test_chain
  // COMMON: d2m.generic
  // Graph coloring allocates 2 slices for inputs; intermediate results (tile_sub, tile_eqz)
  // are computed in registers without DST storage, only final result needs DST
  // COMMON: %[[DST:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst>
  // COMMON: %[[MEM0:.*]] = d2m.wait
  // COMMON: %[[MEM1:.*]] = d2m.wait
  // COMMON: %[[MEM_OUT:.*]] = d2m.reserve
  // COMMON: affine.load %[[MEM0]][%c0, %c0]
  // COMMON: affine.load %[[MEM1]][%c0, %c0]
  // COMMON: "d2m.tile_sub"
  // COMMON: "d2m.tile_eqz"
  // BASIC: "d2m.tile_eqz"{{.*}}{result_dst_index = 1 : i64}
  // CHAITIN: "d2m.tile_eqz"{{.*}}{result_dst_index = 1 : i64}
  // LEGACY: "d2m.tile_eqz"{{.*}}{result_dst_index = 1 : i64}
  // COMMON: affine.store {{.*}}, %[[MEM_OUT]][%c0, %c0]
  // COMMON: d2m.release_dst %[[DST]] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst>

  func.func @test_chain(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
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

// Verifies mixed unary/binary operation chains requiring multiple DST registers.
// Tests 2x2 tile processing where binary operation forces different register allocation strategies.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // COMMON-LABEL: func.func @test_chain_multi_tile
  // COMMON: d2m.generic
  // COMMON: ^compute0(%[[CB0:.*]]: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %[[CB1:.*]]: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %[[CB_OUT:.*]]: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>):

  // Binary operation requires separate DST registers for each input chain
  // Different strategies will allocate different number of slices
  // COMMON: %[[DST:.*]] = d2m.acquire_dst() : memref<{{[0-9]+}}x2x2x!ttcore.tile<32x32, f32>, #dst>
  // COMMON: %[[MEM0:.*]] = d2m.wait %[[CB0]]
  // COMMON: %[[MEM1:.*]] = d2m.wait %[[CB1]]
  // COMMON: %[[MEM_OUT:.*]] = d2m.reserve %[[CB_OUT]]
  // COMMON: affine.for
  // COMMON: affine.load %[[MEM0]][{{.*}}]
  // COMMON: affine.load %[[MEM1]][{{.*}}]
  // COMMON: affine.store {{.*}}, %[[MEM_OUT]][{{.*}}]

  // Verify result_dst_index attribute is attached for all strategies
  // BASIC: "d2m.tile_add"{{.*}}{result_dst_index = 1 : i64}
  // CHAITIN: "d2m.tile_add"{{.*}}{result_dst_index = 1 : i64}
  // GREEDY: "d2m.tile_add"{{.*}}{result_dst_index = 0 : i64}

  func.func @test_chain_multi_tile(%in0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>,
                                   %in1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>,
                                   %out: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
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
    } ins(%in0, %in1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>,
                       memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>)
      outs(%out : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb_out: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          %v0 = affine.load %mem0[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %abs_result = "d2m.tile_abs"(%v0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          %sin_result = "d2m.tile_sin"(%abs_result) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>

          %v1 = affine.load %mem1[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %neg_result = "d2m.tile_negative"(%v1) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>

          %add_result = "d2m.tile_add"(%sin_result, %neg_result) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %add_result, %mem_out[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
    }
    return
  }
}

// -----

// Verifies complex binary operation chains showing strategy differences.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // COMMON-LABEL: func.func @test_multi_chain
  // COMMON: d2m.generic
  // Demonstrates strategy-specific allocation differences
  // COMMON: %[[DST:.*]] = d2m.acquire_dst() : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst>
  // COMMON: affine.load {{.*}}: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  // COMMON: "d2m.tile_abs"
  // COMMON: "d2m.tile_sin"
  // COMMON: "d2m.tile_negative"
  // COMMON: "d2m.tile_add"
  // GREEDY: "d2m.tile_mul"{{.*}}{result_dst_index = 0 : i64}
  // CHAITIN: "d2m.tile_mul"{{.*}}{result_dst_index = 2 : i64}
  // BASIC: "d2m.tile_mul"{{.*}}{result_dst_index = 2 : i64}
  // COMMON: affine.store {{.*}}: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

  func.func @test_multi_chain(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
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

// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=greedy" --split-input-file %s | FileCheck %s --check-prefixes=COMMON,GC
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=chaitin-briggs" --split-input-file %s | FileCheck %s --check-prefixes=COMMON,GC
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access="max-dst-physical-size-tiles=32" --split-input-file %s | FileCheck %s --check-prefixes=COMMON,LEGACY
//
// Tests for graph coloring DST allocation with d2m.linalg_root marked loops.
// These tests use loops with the d2m.linalg_root attribute which both
// the legacy and graph coloring passes recognize.
// The key difference: GC allocates minimal DST slices, LEGACY allocates more conservatively.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // COMMON-LABEL: func.func @test_binary_with_loops
  // COMMON: d2m.generic

  // COMMON: %[[MEM0:.*]] = d2m.wait
  // COMMON: %[[MEM1:.*]] = d2m.wait
  // COMMON: %[[MEM_OUT:.*]] = d2m.reserve

  // GC pass allocates minimal slices (3), LEGACY allocates more conservatively (16 for f16)
  // GC: %[[DST:.*]] = d2m.acquire_dst() : memref<3x1x1x!ttcore.tile<32x32, f16>, #dst>
  // LEGACY: %[[DST:.*]] = d2m.acquire_dst() : memref<16x1x1x!ttcore.tile<32x32, f16>, #dst>

  // Verify prologue, compute, and epilogue loops are generated
  // COMMON: affine.for
  // COMMON: affine.load %[[MEM0]]
  // COMMON: affine.store {{.*}}, %[[DST]]
  // COMMON: affine.load %[[MEM1]]
  // COMMON: affine.store {{.*}}, %[[DST]]

  // COMMON: affine.for
  // COMMON: affine.load %[[DST]]
  // COMMON: affine.load %[[DST]]
  // COMMON: "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}
  // COMMON: affine.store {{.*}}, %[[DST]]

  // COMMON: affine.for
  // COMMON: affine.load %[[DST]]
  // COMMON: affine.store {{.*}}, %[[MEM_OUT]]


  func.func @test_binary_with_loops(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
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
      } {d2m.linalg_root}
    }
    return
  }
}

// -----

// Verifies f32 data type support with d2m.generic wrapper and result_dst_index checks.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // COMMON-LABEL: func.func @test_f32_dtype_with_loops
  // COMMON: d2m.generic
  // COMMON: d2m.wait
  // COMMON: d2m.wait
  // COMMON: d2m.reserve
  // GC: %[[DST:.*]] = d2m.acquire_dst() : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst>
  // LEGACY: %[[DST:.*]] = d2m.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
  // COMMON: affine.for
  // COMMON: affine.load {{.*}}: memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  // GC: affine.store {{.*}}: memref<3x1x1x!ttcore.tile<32x32, f32>, #dst>
  // LEGACY: affine.store {{.*}}: memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
  // COMMON: affine.load {{.*}}: memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  // GC: affine.store {{.*}}: memref<3x1x1x!ttcore.tile<32x32, f32>, #dst>
  // LEGACY: affine.store {{.*}}: memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
  // COMMON: affine.for
  // COMMON: "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}
  // COMMON: affine.for
  // GC: affine.load {{.*}}: memref<3x1x1x!ttcore.tile<32x32, f32>, #dst>
  // LEGACY: affine.load {{.*}}: memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
  // COMMON: affine.store {{.*}}: memref<1x1x!ttcore.tile<32x32, f32>, #l1>

  func.func @test_f32_dtype_with_loops(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
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
      } {d2m.linalg_root}
    }
    return
  }
}

// -----

// Verifies mixed unary/binary operation chains requiring multiple DST registers.
// Tests 2x2 tile processing where binary operation forces different register allocation strategies.
// Uses f16 tiles to fit in DST capacity (f32 tiles take 2x DST space).

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // COMMON-LABEL: func.func @test_2x2_binary_with_nested_loops
  // COMMON: d2m.generic

  // COMMON: %[[MEM0:.*]] = d2m.wait
  // COMMON: %[[MEM1:.*]] = d2m.wait
  // COMMON: %[[MEM_OUT:.*]] = d2m.reserve

  // Binary operation requires separate DST registers for each input chain
  // Graph coloring allocates 3 slices (2 inputs + 1 result), legacy allocates 4 slices
  // GC: %[[DST:.*]] = d2m.acquire_dst() : memref<3x2x2x!ttcore.tile<32x32, f16>, #dst>
  // LEGACY: %[[DST:.*]] = d2m.acquire_dst() : memref<4x2x2x!ttcore.tile<32x32, f16>, #dst>

  // Prologue loop copies inputs to DST
  // COMMON: affine.for
  // COMMON: affine.load %[[MEM0]]
  // COMMON: affine.store {{.*}}, %[[DST]]
  // COMMON: affine.load %[[MEM1]]
  // COMMON: affine.store {{.*}}, %[[DST]]

  // Compute loop with DST operations
  // COMMON: affine.for
  // COMMON: affine.load %[[DST]]
  // COMMON: "d2m.tile_abs"
  // COMMON: "d2m.tile_sin"
  // COMMON: affine.load %[[DST]]
  // COMMON: "d2m.tile_negative"
  // Verify result_dst_index attribute is attached for all strategies
  // LEGACY: "d2m.tile_add"{{.*}}{result_dst_index = 2 : i64}
  // GC: "d2m.tile_add"{{.*}}{result_dst_index = {{[0-9]+}} : i64}
  // COMMON: affine.store {{.*}}, %[[DST]]

  // Epilogue loop copies result back to L1
  // COMMON: affine.for
  // COMMON: affine.load %[[DST]]
  // COMMON: affine.store {{.*}}, %[[MEM_OUT]]


  func.func @test_2x2_binary_with_nested_loops(%in0: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>,
                                   %in1: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>,
                                   %out: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>) {
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
    } ins(%in0, %in1 : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>,
                       memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>)
      outs(%out : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb_out: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f16>, #l1_>

      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          %v0 = affine.load %mem0[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
          %abs_result = "d2m.tile_abs"(%v0) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %sin_result = "d2m.tile_sin"(%abs_result) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>

          %v1 = affine.load %mem1[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
          %neg_result = "d2m.tile_negative"(%v1) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>

          %add_result = "d2m.tile_add"(%sin_result, %neg_result) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          affine.store %add_result, %mem_out[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
        }
      } {d2m.linalg_root}
    }
    return
  }
}

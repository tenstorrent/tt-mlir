// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="max-dst-physical-size-tiles=32" --canonicalize %s | FileCheck %s

// Tests graph coloring DST allocation for matmul operations with large DST capacity.

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @no_loops
  func.func @no_loops(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                      %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                      %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                       memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
      outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK: %[[MEM0:.*]] = d2m.wait
      // CHECK: %[[MEM1:.*]] = d2m.wait
      // CHECK: %[[MEM2:.*]] = d2m.reserve
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

      %c0 = arith.constant 0 : index

      // Load inputs from L1
      // CHECK: %[[A:.*]] = affine.load %[[MEM0]][0, 0]
      // CHECK: %[[B:.*]] = affine.load %[[MEM1]][0, 0]
      %a = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %b = affine.load %mem1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c = affine.load %mem2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

      // DST acquired after input loads
      // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst>
      // Accumulator loaded from L1 and stored to DST
      // CHECK: %[[C:.*]] = affine.load %[[MEM2]][0, 0]
      // CHECK: affine.store %[[C]], %[[DST]][0, 0, 0]
      // Accumulator loaded from DST for matmul
      // CHECK: %[[C_DST:.*]] = affine.load %[[DST]][0, 0, 0]
      // Matmul with A, B from L1 and C from DST
      // CHECK: %[[RESULT:.*]] = "d2m.tile_matmul"(%[[A]], %[[B]], %[[C_DST]])
      %result = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>

      // Result stored to DST, then loaded and written to L1
      // CHECK: affine.store %[[RESULT]], %[[DST]][0, 0, 0]
      // CHECK: %[[FINAL:.*]] = affine.load %[[DST]][0, 0, 0]
      // CHECK: affine.store %[[FINAL]], %[[MEM2]][0, 0]
      affine.store %result, %mem2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

      // CHECK: d2m.release_dst %[[DST]]
    }
    return
  }

  // CHECK-LABEL: func.func @generic_matmul
  func.func @generic_matmul(%in0: memref<1x1x3x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #l1_>,
                            %in1: memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>,
                            %out0: memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1 : memref<1x1x3x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #l1_>,
                       memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>)
      outs(%out0 : memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<3x3x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb2: !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK: %[[MEM0:.*]] = d2m.wait
      // CHECK: %[[MEM1:.*]] = d2m.wait
      // CHECK: %[[MEM2:.*]] = d2m.reserve
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<3x3x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x3x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x2x!ttcore.tile<32x32, f32>, #l1_>

      // DST acquired before loops
      // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>

      // Prologue loop: copy accumulator from L1 to DST
      // CHECK: affine.for
      // CHECK: affine.for
      // CHECK: affine.for
      // CHECK: affine.load %[[MEM2]]
      // CHECK: affine.store {{.*}}, %[[DST]]

      // Main matmul computation loops
      // CHECK: affine.for %[[I:.*]] = 0 to 3
      // CHECK: affine.for %[[J:.*]] = 0 to 2
      // CHECK: affine.for %[[K:.*]] = 0 to 3
      affine.for %i = 0 to 3 {
        affine.for %j = 0 to 2 {
          affine.for %k = 0 to 3 {
            // Load inputs from L1
            // CHECK: %[[A:.*]] = affine.load %[[MEM0]]
            // CHECK: %[[B:.*]] = affine.load %[[MEM1]]
            %a = affine.load %mem0[%i, %k] : memref<3x3x!ttcore.tile<32x32, f32>, #l1_>
            %b = affine.load %mem1[%k, %j] : memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
            %c = affine.load %mem2[%i, %j] : memref<3x2x!ttcore.tile<32x32, f32>, #l1_>

            // Accumulator loaded from DST for matmul
            // CHECK: %[[C_DST:.*]] = affine.load %[[DST]]
            // CHECK: %[[RESULT:.*]] = "d2m.tile_matmul"(%[[A]], %[[B]], %[[C_DST]])
            %result = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>

            // Result stored back to DST
            // CHECK: affine.store %[[RESULT]], %[[DST]]
            affine.store %result, %mem2[%i, %j] : memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
          }
        }
      }

      // Epilogue loop: copy result from DST back to L1
      // CHECK: affine.for
      // CHECK: affine.for
      // CHECK: affine.for
      // CHECK: affine.load %[[DST]]
      // CHECK: affine.store {{.*}}, %[[MEM2]]

      // CHECK: d2m.release_dst %[[DST]]
    }
    return
  }
}

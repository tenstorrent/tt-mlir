// RUN: ttmlir-opt --ttcore-register-device --ttir-insert-dst-register-access --canonicalize %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
module {
  func.func private @no_loops(%arg0: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    // CHECK: %[[DST:.*]] = ttir.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
    %0 = affine.load %arg0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK: %[[ARG2_VAL:.*]] = affine.load %arg2
    %2 = affine.load %arg2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    // Check that the third operand (accumulator) is stored to dst memory space
    // CHECK: affine.store %[[ARG2_VAL]], %[[DST]]
    // Check that the accumulator is loaded back from dst memory space for the matmul
    // CHECK: %[[DST_VAL:.*]] = affine.load %[[DST]]
    // CHECK: %[[MATMUL_RESULT:.*]] = "ttir.tile_matmul"
    %3 = "ttir.tile_matmul"(%0, %1, %2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // Check that matmul result is stored back to dst memory space
    // CHECK: affine.store %[[MATMUL_RESULT]], %[[DST]]
    // Check that result is loaded from dst memory space
    // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]]
    // Check that final result is stored back to original #l1 memory space
    // CHECK: affine.store %[[FINAL_VAL]], %arg2
    affine.store %3, %arg2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  func.func private @binary(%arg0: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    // CHECK: %[[DST:.*]] = ttir.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
    %0 = affine.load %arg0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    // Check that the operands are stored to dst memory space
    // CHECK: %[[ARG0_VAL:.*]] = affine.load %arg0
    // CHECK: affine.store %[[ARG0_VAL]], %[[DST]]
    // CHECK: %[[DST0_VAL:.*]] = affine.load %[[DST]]
    // CHECK: %[[ARG1_VAL:.*]] = affine.load %arg1
    // CHECK: affine.store %[[ARG1_VAL]], %[[DST]]
    // CHECK: %[[DST1_VAL:.*]] = affine.load %[[DST]]
    // CHECK: %[[MAXIMUM_RESULT:.*]] = "ttir.tile_maximum"
    %3 = "ttir.tile_maximum"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // Check that maximum result is stored back to dst memory space
    // CHECK: affine.store %[[MAXIMUM_RESULT]], %[[DST]]
    // Check that result is loaded from dst memory space
    // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]]
    // Check that final result is stored back to original #l1 memory space
    // CHECK: affine.store %[[FINAL_VAL]], %arg2
    affine.store %3, %arg2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  func.func private @generic_matmul(
    %arg0: memref<1x1x3x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #ttcore.memory_space<l1>>,
    %arg1: memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.memory_space<l1>>,
    %arg2: memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.memory_space<l1>>
    ) {
    ttir.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>], threads = [#ttir.thread<compute>]}
        ins(%arg0, %arg1 : memref<1x1x3x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #ttcore.memory_space<l1>>, memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.memory_space<l1>>)
        outs(%arg2 : memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.memory_space<l1>>)  {
    ^compute0(%cb0: memref<3x3x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %cb1: memref<3x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %cb2: memref<3x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>):
      // Check that constants and destination buffer are created
      // CHECK: %[[C0:.*]] = arith.constant 0 : index
      // CHECK: %[[DST:.*]] = ttir.acquire_dst() : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>

      // Check for iteration index and conditional initialization
      // CHECK: %[[ITER2:.*]] = ttir.iter_index(2) : index
      // CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[ITER2]], %[[C0]] : index
      // CHECK: scf.if %[[CMP]] {

      // Check conditional initialization loop structure (2D loop for initialization)
      // CHECK: affine.for %[[INIT_I:.*]] = 0 to 3 {
      // CHECK-NEXT: affine.for %[[INIT_J:.*]] = 0 to 2 {

      // Check initialization: load from l1, store to dst
      // CHECK: %[[INIT_VAL:.*]] = affine.load %cb2[%[[INIT_I]], %[[INIT_J]]] : memref<3x2x!ttcore.tile<32x32, f32>, #l1>
      // CHECK: affine.store %[[INIT_VAL]], %[[DST]][0, %[[INIT_I]], %[[INIT_J]]] : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>

      // Check main computation loop structure (3D loop nest)
      // CHECK: affine.for %[[I:.*]] = 0 to 3 {
      // CHECK-NEXT: affine.for %[[J:.*]] = 0 to 2 {
      // CHECK-NEXT: affine.for %[[K:.*]] = 0 to 3 {

      // Check loads in computation loop: first two from l1, accumulator from dst
      // CHECK: %[[A_VAL:.*]] = affine.load %cb0[%[[I]], %[[K]]] : memref<3x3x!ttcore.tile<32x32, f32>, #l1>
      // CHECK: %[[B_VAL:.*]] = affine.load %cb1[%[[K]], %[[J]]] : memref<3x2x!ttcore.tile<32x32, f32>, #l1>
      // CHECK: %[[C_VAL:.*]] = affine.load %[[DST]][0, %[[I]], %[[J]]] : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>

      // Check matmul operation uses values from correct memory spaces
      // CHECK: %[[MATMUL_RESULT:.*]] = "ttir.tile_matmul"(%[[A_VAL]], %[[B_VAL]], %[[C_VAL]]) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>

      // Check result is stored back to dst
      // CHECK: affine.store %[[MATMUL_RESULT]], %[[DST]][0, %[[I]], %[[J]]] : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>
      affine.for %i2 = 0 to 3 {
        affine.for %i3 = 0 to 2 {
          affine.for %i4 = 0 to 3 {
            %0 = affine.load %cb0[%i2, %i4] : memref<3x3x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
            %1 = affine.load %cb1[%i4, %i3] : memref<3x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
            %2 = affine.load %cb2[%i2, %i3] : memref<3x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
            %3 = "ttir.tile_matmul"(%0, %1, %2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            affine.store %3, %cb2[%i2, %i3] : memref<3x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
          }
        }
      }

      // Check final writeback loop structure (2D loop)
      // CHECK: affine.for %[[WB_I:.*]] = 0 to 3 {
      // CHECK-NEXT: affine.for %[[WB_J:.*]] = 0 to 2 {

      // Check writeback: load from dst, store to l1
      // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]][0, %[[WB_I]], %[[WB_J]]] : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>
      // CHECK: affine.store %[[FINAL_VAL]], %cb2[%[[WB_I]], %[[WB_J]]] : memref<3x2x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }
}

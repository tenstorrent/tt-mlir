// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc --split-input-file %s | FileCheck %s
//
// Verifies that d2m-insert-dst-register-gc correctly handles multiple d2m.generic
// operations in the same function. Each generic should get its own DST allocation.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @multiple_generics_in_function
  func.func @multiple_generics_in_function(
    %in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
    %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
    %temp: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
    %out: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>
  ) {
    // First generic: adds two inputs and stores to temp
    // CHECK: d2m.generic
    // CHECK: d2m.wait
    // CHECK: d2m.wait
    // CHECK: d2m.reserve
    // CHECK: %[[DST0:.*]] = d2m.acquire_dst() : memref<3x1x1x!ttcore.tile<32x32, f16>, #dst>
    // L1 -> DST copies
    // CHECK: affine.store {{.*}}, %[[DST0]][0,
    // CHECK: affine.store {{.*}}, %[[DST0]][1,
    // Compute operation using DST values
    // CHECK: "d2m.tile_add"
    // Result stored to DST slice 2
    // CHECK: affine.store {{.*}}, %[[DST0]]
    // CHECK: d2m.release_dst %[[DST0]]
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
      outs(%temp : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
      %c0 = arith.constant 0 : index
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>

      %v0 = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %v1 = affine.load %mem1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %add_result = "d2m.tile_add"(%v0, %v1) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
      affine.store %add_result, %mem_out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
    }

    // Second generic: multiplies temp with in0 and stores to output
    // CHECK: d2m.generic
    // CHECK: d2m.wait
    // CHECK: d2m.wait
    // CHECK: d2m.reserve
    // CHECK: %[[DST1:.*]] = d2m.acquire_dst() : memref<3x1x1x!ttcore.tile<32x32, f16>, #dst>
    // L1 -> DST copies
    // CHECK: affine.store {{.*}}, %[[DST1]][0,
    // CHECK: affine.store {{.*}}, %[[DST1]][1,
    // Compute operation using DST values
    // CHECK: "d2m.tile_mul"
    // Result stored to DST slice 2
    // CHECK: affine.store {{.*}}, %[[DST1]]
    // CHECK: d2m.release_dst %[[DST1]]
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
    } ins(%temp, %in0 :
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute1(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
      %c0 = arith.constant 0 : index
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>

      %v0 = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %v1 = affine.load %mem1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mul_result = "d2m.tile_mul"(%v0, %v1) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
      affine.store %mul_result, %mem_out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
    }

    return
  }
}

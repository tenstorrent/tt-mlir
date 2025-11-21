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
    // CHECK: %[[DST0:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>
    // CHECK-DAG: %[[C0_0:.*]] = arith.constant 0
    // CHECK-DAG: %[[MEM0_0:.*]] = d2m.wait
    // CHECK-DAG: %[[MEM1_0:.*]] = d2m.wait
    // CHECK-DAG: %[[MEMOUT_0:.*]] = d2m.reserve
    // L1 -> DST copies (may be reordered)
    // CHECK-DAG: %[[V0_L1:.*]] = affine.load %[[MEM0_0]][%[[C0_0]], %[[C0_0]]]
    // CHECK-DAG: affine.store %[[V0_L1]], %[[DST0]][{{.*}}, %[[C0_0]], %[[C0_0]]]
    // CHECK-DAG: %[[V1_L1:.*]] = affine.load %[[MEM1_0]][%[[C0_0]], %[[C0_0]]]
    // CHECK-DAG: affine.store %[[V1_L1]], %[[DST0]][{{.*}}, %[[C0_0]], %[[C0_0]]]
    // DST -> operation operands (may be reordered)
    // CHECK-DAG: %[[V0_DST:.*]] = affine.load %[[DST0]][{{.*}}, %[[C0_0]], %[[C0_0]]]
    // CHECK-DAG: %[[V1_DST:.*]] = affine.load %[[DST0]][{{.*}}, %[[C0_0]], %[[C0_0]]]
    // Compute operation using DST values
    // CHECK: %[[ADD_RESULT:.*]] = "d2m.tile_add"(%[[V0_DST]], %[[V1_DST]])
    // Result -> DST -> L1
    // CHECK: affine.store %[[ADD_RESULT]], %[[DST0]][{{.*}}, %[[C0_0]], %[[C0_0]]]
    // CHECK: %[[RESULT_DST:.*]] = affine.load %[[DST0]][{{.*}}, %[[C0_0]], %[[C0_0]]]
    // CHECK: affine.store %[[RESULT_DST]], %[[MEMOUT_0]][%[[C0_0]], %[[C0_0]]]
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
    // CHECK: %[[DST1:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>
    // CHECK-DAG: %[[C0_1:.*]] = arith.constant 0
    // CHECK-DAG: %[[MEM0_1:.*]] = d2m.wait
    // CHECK-DAG: %[[MEM1_1:.*]] = d2m.wait
    // CHECK-DAG: %[[MEMOUT_1:.*]] = d2m.reserve
    // L1 -> DST copies (may be reordered)
    // CHECK-DAG: %[[TEMP_L1:.*]] = affine.load %[[MEM0_1]][%[[C0_1]], %[[C0_1]]]
    // CHECK-DAG: affine.store %[[TEMP_L1]], %[[DST1]][{{.*}}, %[[C0_1]], %[[C0_1]]]
    // CHECK-DAG: %[[IN0_L1:.*]] = affine.load %[[MEM1_1]][%[[C0_1]], %[[C0_1]]]
    // CHECK-DAG: affine.store %[[IN0_L1]], %[[DST1]][{{.*}}, %[[C0_1]], %[[C0_1]]]
    // DST -> operation operands (may be reordered)
    // CHECK-DAG: %[[TEMP_DST:.*]] = affine.load %[[DST1]][{{.*}}, %[[C0_1]], %[[C0_1]]]
    // CHECK-DAG: %[[IN0_DST:.*]] = affine.load %[[DST1]][{{.*}}, %[[C0_1]], %[[C0_1]]]
    // Compute operation using DST values
    // CHECK: %[[MUL_RESULT:.*]] = "d2m.tile_mul"(%[[TEMP_DST]], %[[IN0_DST]])
    // Result -> DST -> L1
    // CHECK: affine.store %[[MUL_RESULT]], %[[DST1]][{{.*}}, %[[C0_1]], %[[C0_1]]]
    // CHECK: %[[FINAL_DST:.*]] = affine.load %[[DST1]][{{.*}}, %[[C0_1]], %[[C0_1]]]
    // CHECK: affine.store %[[FINAL_DST]], %[[MEMOUT_1]][%[[C0_1]], %[[C0_1]]]
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

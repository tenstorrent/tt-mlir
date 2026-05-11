// RUN: ttmlir-opt --split-input-file \
// RUN:   --d2m-distribute-compute-threads \
// RUN:   --d2m-materialize-compute-thread-forall %s | FileCheck %s

// Smoke test: distribute followed by materialize composes correctly.
// After both passes:
//   - no scf.forall remains
//   - one d2m.my_thread_id appears (inside the d2m.generic compute region)
//   - the inner linalg.generic operates on memref.subview ops indexed by
//     the my_thread_id result.

// -----

func.func @smoke(
    %A: memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %B: memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %C: memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
  // CHECK-LABEL: func.func @smoke
  // CHECK: d2m.generic
  // CHECK-NOT: scf.forall
  // CHECK: %[[TID:.*]] = d2m.my_thread_id : index
  // CHECK: affine.apply {{.*}}(%[[TID]])
  // CHECK: memref.subview
  // CHECK: linalg.generic
  // CHECK: d2m.tile_matmul
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
      ins(%A, %B : memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%C : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%C : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>, %c: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
  }
  return
}

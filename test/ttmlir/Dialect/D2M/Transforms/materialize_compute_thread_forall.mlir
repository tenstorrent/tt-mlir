// RUN: ttmlir-opt --split-input-file --allow-unregistered-dialect \
// RUN:   --d2m-materialize-compute-thread-forall %s | FileCheck %s

// -----

// Basic case: scf.forall with #d2m.compute_thread mapping inside a
// d2m.generic Compute thread region is replaced by %tid = d2m.my_thread_id ;
// <inlined body using %tid>. The body ops appear at the parent-block level
// in their original order with the IV-use rewired through the my_thread_id
// result. The affine map and subview shape/strides are propagated unchanged.
// The enclosing d2m.generic's Compute ThreadAttr is rewritten to carry
// num_compute_threads = 4 (extracted from #d2m.compute_thread<num=4>).

// CHECK: #[[$MAP:[^ ]+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: func.func @materialize_basic
// CHECK-SAME:  (%[[A:[A-Za-z0-9_]+]]: memref<8x8xf32>)
// CHECK:         d2m.generic
// CHECK-SAME:    threads = [#d2m.thread<compute, num_compute_threads = 4>]
// CHECK:         %[[TID:[A-Za-z0-9_]+]] = d2m.my_thread_id : index
// CHECK-NEXT:    %[[OFF:[A-Za-z0-9_]+]] = affine.apply #[[$MAP]](%[[TID]])
// CHECK-NEXT:    %[[SUB:[A-Za-z0-9_]+]] = memref.subview %[[A]][%[[OFF]], 0] [2, 8] [1, 1] : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
// CHECK-NEXT:    "use"(%[[SUB]]) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
// CHECK-NOT: scf.forall
// CHECK-NOT: scf.in_parallel
// CHECK-NOT: #d2m.compute_thread

func.func @materialize_basic(%A: memref<8x8xf32>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute>]}
      ins()
      outs(%A : memref<8x8xf32>) {
    scf.forall (%tid) in (4) {
      %off = affine.apply affine_map<(d0) -> (d0 * 2)>(%tid)
      %sub = memref.subview %A[%off, 0][2, 8][1, 1]
        : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
      "use"(%sub) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
    } {mapping = [#d2m.compute_thread<num = 4>]}
  }
  return
}

// -----

// Multiple compute-thread foralls in separate d2m.generics are both lowered.
// Each forall gets its own my_thread_id (TID0, TID1) inside its own generic,
// and each generic's Compute ThreadAttr picks up num_compute_threads = 4
// independently.

// CHECK: #[[$MAP:[^ ]+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: func.func @materialize_multiple
// CHECK-SAME:  (%[[A:[A-Za-z0-9_]+]]: memref<8x8xf32>, %[[B:[A-Za-z0-9_]+]]: memref<8x8xf32>)
// CHECK:         d2m.generic
// CHECK-SAME:    threads = [#d2m.thread<compute, num_compute_threads = 4>]
// CHECK:         %[[TID0:[A-Za-z0-9_]+]] = d2m.my_thread_id : index
// CHECK-NEXT:    %[[OFF0:[A-Za-z0-9_]+]] = affine.apply #[[$MAP]](%[[TID0]])
// CHECK-NEXT:    %[[S0:[A-Za-z0-9_]+]] = memref.subview %[[A]][%[[OFF0]], 0] [2, 8] [1, 1] : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
// CHECK-NEXT:    "use_a"(%[[S0]]) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
// CHECK:         d2m.generic
// CHECK-SAME:    threads = [#d2m.thread<compute, num_compute_threads = 4>]
// CHECK:         %[[TID1:[A-Za-z0-9_]+]] = d2m.my_thread_id : index
// CHECK-NEXT:    %[[OFF1:[A-Za-z0-9_]+]] = affine.apply #[[$MAP]](%[[TID1]])
// CHECK-NEXT:    %[[S1:[A-Za-z0-9_]+]] = memref.subview %[[B]][%[[OFF1]], 0] [2, 8] [1, 1] : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
// CHECK-NEXT:    "use_b"(%[[S1]]) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
// CHECK-NOT: scf.forall
// CHECK-NOT: scf.in_parallel
// CHECK-NOT: #d2m.compute_thread

func.func @materialize_multiple(%A: memref<8x8xf32>, %B: memref<8x8xf32>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute>]}
      ins()
      outs(%A : memref<8x8xf32>) {
    scf.forall (%tid0) in (4) {
      %off0 = affine.apply affine_map<(d0) -> (d0 * 2)>(%tid0)
      %s0 = memref.subview %A[%off0, 0][2, 8][1, 1]
        : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
      "use_a"(%s0) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
    } {mapping = [#d2m.compute_thread<num = 4>]}
  }
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute>]}
      ins()
      outs(%B : memref<8x8xf32>) {
    scf.forall (%tid1) in (4) {
      %off1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%tid1)
      %s1 = memref.subview %B[%off1, 0][2, 8][1, 1]
        : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
      "use_b"(%s1) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
    } {mapping = [#d2m.compute_thread<num = 4>]}
  }
  return
}

// -----

// Rank-2 compute-thread forall is lowered using one d2m.my_thread_id and two
// derived coordinates. The flattened thread id is decoded as
// row = tid floordiv 2, col = tid mod 2 for a 2x2 forall.

// CHECK-LABEL: func.func @materialize_rank2
// CHECK-SAME:  (%[[A:[A-Za-z0-9_]+]]: memref<2x2xf32>)
// CHECK:         d2m.generic
// CHECK-SAME:    threads = [#d2m.thread<compute, num_compute_threads = 4>]
// CHECK:         %[[TID:[A-Za-z0-9_]+]] = d2m.my_thread_id : index
// CHECK-NEXT:    %[[ROW:[A-Za-z0-9_]+]] = affine.apply {{.*}}(%[[TID]])
// CHECK-NEXT:    %[[COL:[A-Za-z0-9_]+]] = affine.apply {{.*}}(%[[TID]])
// CHECK-NEXT:    "use_rank2"(%[[ROW]], %[[COL]]) : (index, index) -> ()
// CHECK-NOT: scf.forall

func.func @materialize_rank2(%A: memref<2x2xf32>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute>]}
      ins()
      outs(%A : memref<2x2xf32>) {
    scf.forall (%tid0, %tid1) in (2, 2) {
      "use_rank2"(%tid0, %tid1) : (index, index) -> ()
    } {mapping = [#d2m.compute_thread<num = 2>, #d2m.compute_thread<num = 2>]}
  }
  return
}

// -----

// Rank-1 foralls with a 2-way fallback split still materialize and stamp the
// exact thread count used by the mapping.

// CHECK-LABEL: func.func @materialize_factor2
// CHECK-SAME:  (%[[A:[A-Za-z0-9_]+]]: memref<2x8xf32>)
// CHECK:         d2m.generic
// CHECK-SAME:    threads = [#d2m.thread<compute, num_compute_threads = 2>]
// CHECK:         %[[TID:[A-Za-z0-9_]+]] = d2m.my_thread_id : index
// CHECK-NEXT:    "use_factor2"(%[[TID]]) : (index) -> ()
// CHECK-NOT: scf.forall

func.func @materialize_factor2(%A: memref<2x8xf32>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute>]}
      ins()
      outs(%A : memref<2x8xf32>) {
    scf.forall (%tid) in (2) {
      "use_factor2"(%tid) : (index) -> ()
    } {mapping = [#d2m.compute_thread<num = 2>]}
  }
  return
}

// -----

// Rank-1 foralls with a 3-way fallback split still materialize and stamp the
// exact thread count used by the mapping.

// CHECK-LABEL: func.func @materialize_factor3
// CHECK-SAME:  (%[[A:[A-Za-z0-9_]+]]: memref<3x8xf32>)
// CHECK:         d2m.generic
// CHECK-SAME:    threads = [#d2m.thread<compute, num_compute_threads = 3>]
// CHECK:         %[[TID:[A-Za-z0-9_]+]] = d2m.my_thread_id : index
// CHECK-NEXT:    "use_factor3"(%[[TID]]) : (index) -> ()
// CHECK-NOT: scf.forall

func.func @materialize_factor3(%A: memref<3x8xf32>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute>]}
      ins()
      outs(%A : memref<3x8xf32>) {
    scf.forall (%tid) in (3) {
      "use_factor3"(%tid) : (index) -> ()
    } {mapping = [#d2m.compute_thread<num = 3>]}
  }
  return
}

// -----

// scf.forall WITHOUT a #d2m.compute_thread mapping is left completely
// untouched: same body, same IV use, no my_thread_id inserted anywhere.
// The unrelated forall is allowed to live outside any d2m.generic — the
// pass only inspects foralls that carry a #d2m.compute_thread mapping.

// CHECK-LABEL: func.func @leave_unrelated_forall
// CHECK-SAME:  (%[[A:[A-Za-z0-9_]+]]: memref<8x8xf32>)
// CHECK-NEXT:    scf.forall (%[[TID:[A-Za-z0-9_]+]]) in (4) {
// CHECK-NEXT:      %[[OFF:[A-Za-z0-9_]+]] = affine.apply {{.*}}(%[[TID]])
// CHECK-NEXT:      %[[SUB:[A-Za-z0-9_]+]] = memref.subview %[[A]][%[[OFF]], 0] [2, 8] [1, 1] : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
// CHECK-NEXT:      "use"(%[[SUB]]) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NOT: d2m.my_thread_id

func.func @leave_unrelated_forall(%A: memref<8x8xf32>) {
  scf.forall (%tid) in (4) {
    %off = affine.apply affine_map<(d0) -> (d0 * 2)>(%tid)
    %sub = memref.subview %A[%off, 0][2, 8][1, 1]
      : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
    "use"(%sub) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
  }
  return
}

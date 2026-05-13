// RUN: ttmlir-opt --split-input-file --allow-unregistered-dialect \
// RUN:   --d2m-materialize-compute-thread-forall %s | FileCheck %s

// -----

// Basic case: scf.forall with #d2m.compute_thread mapping is replaced by
// %tid = d2m.my_thread_id ; <inlined body using %tid>. The body ops appear
// at the parent-block level in their original order with the IV-use rewired
// through the my_thread_id result. The affine map and subview shape/strides
// are propagated from the input unchanged.

// CHECK: #[[$MAP:[^ ]+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: func.func @materialize_basic
// CHECK-SAME:  (%[[A:[A-Za-z0-9_]+]]: memref<8x8xf32>)
// CHECK-NEXT:    %[[TID:[A-Za-z0-9_]+]] = d2m.my_thread_id : index
// CHECK-NEXT:    %[[OFF:[A-Za-z0-9_]+]] = affine.apply #[[$MAP]](%[[TID]])
// CHECK-NEXT:    %[[SUB:[A-Za-z0-9_]+]] = memref.subview %[[A]][%[[OFF]], 0] [2, 8] [1, 1] : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
// CHECK-NEXT:    "use"(%[[SUB]]) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
// CHECK-NEXT:    return
// CHECK-NOT: scf.forall
// CHECK-NOT: scf.in_parallel
// CHECK-NOT: #d2m.compute_thread

func.func @materialize_basic(%A: memref<8x8xf32>) {
  scf.forall (%tid) in (4) {
    %off = affine.apply affine_map<(d0) -> (d0 * 2)>(%tid)
    %sub = memref.subview %A[%off, 0][2, 8][1, 1]
      : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
    "use"(%sub) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
  } {mapping = [#d2m.compute_thread<num = 4>]}
  return
}

// -----

// Multiple compute-thread foralls in the same function are both lowered into
// the parent block in their original order. Each forall gets its own
// my_thread_id (TID0, TID1) and its body's IV-uses are rewired only through
// its own my_thread_id — proven here by binding TID0/TID1 separately and
// checking each affine.apply consumes the matching one.

// CHECK: #[[$MAP:[^ ]+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: func.func @materialize_multiple
// CHECK-SAME:  (%[[A:[A-Za-z0-9_]+]]: memref<8x8xf32>, %[[B:[A-Za-z0-9_]+]]: memref<8x8xf32>)
// CHECK-NEXT:    %[[TID0:[A-Za-z0-9_]+]] = d2m.my_thread_id : index
// CHECK-NEXT:    %[[OFF0:[A-Za-z0-9_]+]] = affine.apply #[[$MAP]](%[[TID0]])
// CHECK-NEXT:    %[[S0:[A-Za-z0-9_]+]] = memref.subview %[[A]][%[[OFF0]], 0] [2, 8] [1, 1] : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
// CHECK-NEXT:    "use_a"(%[[S0]]) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
// CHECK-NEXT:    %[[TID1:[A-Za-z0-9_]+]] = d2m.my_thread_id : index
// CHECK-NEXT:    %[[OFF1:[A-Za-z0-9_]+]] = affine.apply #[[$MAP]](%[[TID1]])
// CHECK-NEXT:    %[[S1:[A-Za-z0-9_]+]] = memref.subview %[[B]][%[[OFF1]], 0] [2, 8] [1, 1] : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
// CHECK-NEXT:    "use_b"(%[[S1]]) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
// CHECK-NEXT:    return
// CHECK-NOT: scf.forall
// CHECK-NOT: scf.in_parallel
// CHECK-NOT: #d2m.compute_thread

func.func @materialize_multiple(%A: memref<8x8xf32>, %B: memref<8x8xf32>) {
  scf.forall (%tid0) in (4) {
    %off0 = affine.apply affine_map<(d0) -> (d0 * 2)>(%tid0)
    %s0 = memref.subview %A[%off0, 0][2, 8][1, 1]
      : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
    "use_a"(%s0) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
  } {mapping = [#d2m.compute_thread<num = 4>]}
  scf.forall (%tid1) in (4) {
    %off1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%tid1)
    %s1 = memref.subview %B[%off1, 0][2, 8][1, 1]
      : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
    "use_b"(%s1) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
  } {mapping = [#d2m.compute_thread<num = 4>]}
  return
}

// -----

// scf.forall WITHOUT a #d2m.compute_thread mapping is left completely
// untouched: same body, same IV use, no my_thread_id inserted anywhere.

// CHECK: #[[$MAP:[^ ]+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: func.func @leave_unrelated_forall
// CHECK-SAME:  (%[[A:[A-Za-z0-9_]+]]: memref<8x8xf32>)
// CHECK-NEXT:    scf.forall (%[[TID:[A-Za-z0-9_]+]]) in (4) {
// CHECK-NEXT:      %[[OFF:[A-Za-z0-9_]+]] = affine.apply #[[$MAP]](%[[TID]])
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

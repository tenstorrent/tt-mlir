// RUN: ttmlir-opt --split-input-file --allow-unregistered-dialect \
// RUN:   --d2m-materialize-compute-thread-forall %s | FileCheck %s

// -----

// Basic case: scf.forall with #d2m.compute_thread mapping is replaced by
// %tid = d2m.my_thread_id ; <inlined body using %tid>.

func.func @materialize_basic(%A: memref<8x8xf32>) {
  // CHECK-LABEL: func.func @materialize_basic
  // CHECK-NOT: scf.forall
  // CHECK: %[[TID:.*]] = d2m.my_thread_id : index
  // CHECK: %[[OFF:.*]] = affine.apply {{.*}}(%[[TID]])
  // CHECK: %[[SUB:.*]] = memref.subview %{{.*}}[%[[OFF]], 0]
  // CHECK: "use"(%[[SUB]])
  scf.forall (%tid) in (4) {
    %off = affine.apply affine_map<(d0) -> (d0 * 2)>(%tid)
    %sub = memref.subview %A[%off, 0][2, 8][1, 1]
      : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
    "use"(%sub) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
  } {mapping = [#d2m.compute_thread<num = 4>]}
  return
}

// -----

// Multiple compute-thread foralls in the same function are both lowered;
// the second lowering inserts a second my_thread_id (one per forall).

func.func @materialize_multiple(%A: memref<8x8xf32>, %B: memref<8x8xf32>) {
  // CHECK-LABEL: func.func @materialize_multiple
  // CHECK-NOT: scf.forall
  // CHECK: d2m.my_thread_id
  // CHECK: "use_a"
  // CHECK: d2m.my_thread_id
  // CHECK: "use_b"
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

// scf.forall WITHOUT a #d2m.compute_thread mapping is left alone — this
// pass only consumes foralls flagged with our specific mapping attribute.

func.func @leave_unrelated_forall(%A: memref<8x8xf32>) {
  // CHECK-LABEL: func.func @leave_unrelated_forall
  // CHECK: scf.forall
  // CHECK-NOT: d2m.my_thread_id
  scf.forall (%tid) in (4) {
    %off = affine.apply affine_map<(d0) -> (d0 * 2)>(%tid)
    %sub = memref.subview %A[%off, 0][2, 8][1, 1]
      : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
    "use"(%sub) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
  }
  return
}

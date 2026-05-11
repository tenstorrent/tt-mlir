// RUN: ttmlir-opt --allow-unregistered-dialect %s | FileCheck %s

// Round-trip test for #d2m.compute_thread<num=N> mapping attribute on scf.forall.

func.func @forall_with_compute_thread_mapping(%A: memref<8x8xf32>) {
  // CHECK: scf.forall (%[[TID:.*]]) in (4)
  // CHECK: } {mapping = [#d2m.compute_thread<num = 4>]}
  scf.forall (%tid) in (4) {
    %off = affine.apply affine_map<(d0) -> (d0 * 2)>(%tid)
    %sub = memref.subview %A[%off, 0][2, 8][1, 1] : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
    "use"(%sub) : (memref<2x8xf32, strided<[8, 1], offset: ?>>) -> ()
  } {mapping = [#d2m.compute_thread<num = 4>]}
  return
}

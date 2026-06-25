// RUN: ttmlir-opt --d2m-affine-licm %s | FileCheck %s

func.func @module_licm() {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %0 = arith.addi %c1, %c2 : index
    }
  }
  return
}

// CHECK-LABEL: func.func @module_licm
// CHECK: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %{{.+}} = arith.addi %[[C1]], %[[C2]] : index
// CHECK: affine.for
// CHECK-NOT: arith.constant
// CHECK-NOT: arith.addi
// CHECK: affine.for
// CHECK-NOT: arith.constant
// CHECK-NOT: arith.addi
// CHECK: return

func.func @module_licm_mixed_loop_types() {
  affine.parallel (%i) = (0) to (4) {
    affine.for %j = 0 to 4 {
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %0 = arith.addi %c3, %c4 : index
    }
    affine.yield
  }
  return
}

// CHECK-LABEL: func.func @module_licm_mixed_loop_types
// CHECK: %[[C3:.+]] = arith.constant 3 : index
// CHECK: %[[C4:.+]] = arith.constant 4 : index
// CHECK: %{{.+}} = arith.addi %[[C3]], %[[C4]] : index
// CHECK: affine.for
// CHECK-NOT: arith.constant
// CHECK-NOT: arith.addi
// CHECK: affine.parallel
// CHECK-NOT: arith.constant
// CHECK-NOT: arith.addi
// CHECK: return

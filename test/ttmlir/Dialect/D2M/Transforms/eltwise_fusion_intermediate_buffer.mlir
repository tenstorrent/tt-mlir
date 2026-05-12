// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --canonicalize --d2m-lower-to-layout --d2m-materialize-view-returns --d2m-generic-fusion %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --canonicalize --d2m-lower-to-layout --d2m-materialize-view-returns --d2m-generic-fusion %s | FileCheck %s --check-prefix=UNIQUE12
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --canonicalize --d2m-lower-to-layout --d2m-materialize-view-returns --d2m-generic-fusion %s | FileCheck %s --check-prefix=UNIQUE23

module {
  func.func @three_op_chain(
      %arg0: tensor<32x32xbf16>,
      %arg1: tensor<32x32xbf16>,
      %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.multiply"(%0, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.sigmoid"(%1) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }
}

// (1) Structural check: the chain fuses into a SINGLE compute d2m.generic.
//     Each consecutive pair of linalg.generic ops must not be separated by
//     another d2m.generic (which would mean the chain wasn't fused), and the
//     three tile ops must appear in original chain order tile_add ->
//     tile_mul -> tile_sigmoid. Trailing CHECK-NOTs forbid any extra
//     linalg.generic / tile op anywhere later in the function.
// CHECK-LABEL: func.func @three_op_chain
// CHECK:           linalg.generic
// CHECK:           "d2m.tile_add"
// CHECK-NOT:       d2m.generic
// CHECK:           linalg.generic
// CHECK:           "d2m.tile_mul"
// CHECK-NOT:       d2m.generic
// CHECK:           linalg.generic
// CHECK:           "d2m.tile_sigmoid"
// CHECK-NOT:       linalg.generic
// CHECK-NOT:       "d2m.tile_add"
// CHECK-NOT:       "d2m.tile_mul"
// CHECK-NOT:       "d2m.tile_sigmoid"

// (2) Aliasing check: each linalg.generic in the fused region must have its
//     OWN outs tensor.empty. We use two separate FileCheck passes to assert
//     pairwise uniqueness across all three linalg.generic ops.
//
//     The first pass (prefix UNIQUE12) captures the first linalg.generic's
//     outs SSA value and CHECK-NOT-scans to end-of-file for any reuse. This
//     rules out first == second AND first == third.
//
//     The second pass (prefix UNIQUE23) skips the first linalg.generic,
//     captures the second linalg.generic's outs SSA value and CHECK-NOT-scans
//     to end-of-file for any reuse. This rules out second == third.
//
//     Together the two passes prove no two of the three linalg.generic ops
//     share the same outs SSA value. Without the fix, all three linalgs share
//     the same outs and both passes fail.
// UNIQUE12-LABEL: func.func @three_op_chain
// UNIQUE12:       linalg.generic{{.*}}outs(%[[FIRST_OUT:[^ ]+]] : tensor
// UNIQUE12-NOT:   linalg.generic{{.*}}outs(%[[FIRST_OUT]] : tensor

// UNIQUE23-LABEL: func.func @three_op_chain
// UNIQUE23:       linalg.generic
// UNIQUE23:       linalg.generic{{.*}}outs(%[[SECOND_OUT:[^ ]+]] : tensor
// UNIQUE23-NOT:   linalg.generic{{.*}}outs(%[[SECOND_OUT]] : tensor

// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// trunc rounds toward zero, so it is idempotent: trunc(trunc(x)) = trunc(x).
// Back-to-back truncs of the same type must fold to a single trunc.
module {
  // CHECK-LABEL: func.func @trunc_idempotence_two_in_a_row
  func.func @trunc_idempotence_two_in_a_row(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK: "ttir.trunc"
    // CHECK-NOT: "ttir.trunc"
    %1 = "ttir.trunc"(%arg0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = "ttir.trunc"(%1) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %2 : tensor<64x64xf32>
  }

  // Differing operand/result types must NOT fold.
  // CHECK-LABEL: func.func @trunc_not_idempotent_different_types
  func.func @trunc_not_idempotent_different_types(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK: "ttir.trunc"
    // CHECK: "ttir.trunc"
    %1 = "ttir.trunc"(%arg0) : (tensor<64x64xf32>) -> tensor<64x64xbf16>
    %2 = "ttir.trunc"(%1) : (tensor<64x64xbf16>) -> tensor<64x64xf32>
    return %2 : tensor<64x64xf32>
  }
}

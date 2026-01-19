// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns --d2m-grid-selection -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test for higher-rank matmul support (issue #6648)
// This test verifies that batch dimensions are preserved during TTIRToD2M transformation

!lhs_2d = tensor<128x96xf32>
!rhs_2d = tensor<96x64xf32>
!result_2d = tensor<128x64xf32>

!lhs_3d = tensor<8x128x96xf32>
!rhs_3d = tensor<8x96x64xf32>
!result_3d = tensor<8x128x64xf32>

!lhs_4d = tensor<32x8x1x128xf32>
!rhs_4d = tensor<32x8x128x128xf32>
!result_4d = tensor<32x8x1x128xf32>

module {
  // CHECK-LABEL: func @matmul_2d
  func.func @matmul_2d(%lhs: !lhs_2d, %rhs: !rhs_2d) -> (!result_2d) {
    // Verify 2D matmul still works (backward compatibility)
    // CHECK: d2m.generic
    // CHECK-SAME: iterator_types = [#parallel, #parallel, #reduction]
    // CHECK: linalg.generic
    // CHECK: d2m.tile_matmul
    %r = "ttir.matmul"(%lhs, %rhs) : (!lhs_2d, !rhs_2d) -> (!result_2d)
    return %r : !result_2d
  }

  // CHECK-LABEL: func @matmul_3d_batched
  func.func @matmul_3d_batched(%lhs: !lhs_3d, %rhs: !rhs_3d) -> (!result_3d) {
    // Verify 3D batched matmul preserves batch dimension
    // Iterator types should be: [batch(parallel), M(parallel), N(parallel), K(reduction)]
    // CHECK: d2m.generic
    // CHECK-SAME: iterator_types = [#parallel, #parallel, #parallel, #reduction]
    // CHECK: linalg.generic
    // CHECK: d2m.tile_matmul
    %r = "ttir.matmul"(%lhs, %rhs) : (!lhs_3d, !rhs_3d) -> (!result_3d)
    return %r : !result_3d
  }

  // CHECK-LABEL: func @matmul_4d_batched
  func.func @matmul_4d_batched(%lhs: !lhs_4d, %rhs: !rhs_4d) -> (!result_4d) {
    // Verify 4D batched matmul preserves both batch dimensions
    // This is the original issue #6648 case: [32,8,1,128] x [32,8,128,128]
    // Iterator types should be: [batch0(parallel), batch1(parallel), M(parallel), N(parallel), K(reduction)]
    // CHECK: d2m.generic
    // CHECK-SAME: iterator_types = [#parallel, #parallel, #parallel, #parallel, #reduction]
    // CHECK: linalg.generic
    // CHECK: d2m.tile_matmul
    %r = "ttir.matmul"(%lhs, %rhs) : (!lhs_4d, !rhs_4d) -> (!result_4d)
    return %r : !result_4d
  }
}

// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="use-tile-matmul=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Test that ttir.matmul with transpose_b=true propagates the transpose flag
// through d2m.tile_matmul -> d2m.tile_matmul_block -> ttkernel::matmul_block.

!lhs = tensor<128x96xf32>
!rhs = tensor<64x96xf32>
!matmul_result = tensor<128x64xf32>

module {
  // CHECK-LABEL: func.func @matmul_transpose_b
  func.func @matmul_transpose_b(%lhs: !lhs, %rhs: !rhs) -> (!matmul_result) {
    // CHECK-NOT: ttir.matmul
    // The kernel-side transpose arg is a compile-time 1 (i.e. true).
    // CHECK: "emitc.constant"() <{value = 1 : i32}>
    // CHECK: mm_block_init
    // CHECK: mm_block_init_short
    // CHECK: matmul_block
    %r = "ttir.matmul"(%lhs, %rhs) <{transpose_a = false, transpose_b = true}> : (!lhs, !rhs) -> (!matmul_result)
    return %r : !matmul_result
  }
}

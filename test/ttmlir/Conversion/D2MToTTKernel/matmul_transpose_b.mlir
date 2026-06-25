// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="use-tile-matmul=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: not ttmlir-opt --ttir-to-ttmetal-pipeline="use-tile-matmul=true" -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=TILE-ERR

// Test that ttir.matmul with transpose_b=true propagates the transpose flag
// through d2m.tile_matmul_block -> ttkernel::matmul_block. The raw
// tile_matmul path does not support transpose_b and must reject it.

!lhs = tensor<128x96xf32>
!rhs = tensor<64x96xf32>
!matmul_result = tensor<128x64xf32>

module {
  // CHECK-LABEL: func.func @matmul_transpose_b
  func.func @matmul_transpose_b(%lhs: !lhs, %rhs: !rhs) -> (!matmul_result) {
    // CHECK-NOT: ttir.matmul
    // The kernel-side transpose arg is a compile-time 1 (i.e. true).
    // CHECK: "emitc.constant"() <{value = 1 : i32}>
    // CHECK: compute_kernel_hw_startup
    // CHECK: matmul_block_init
    // CHECK: experimental::matmul_block
    // TILE-ERR: 'd2m.tile_matmul' op transpose_b is only supported by tile_matmul_block lowering
    %r = "ttir.matmul"(%lhs, %rhs) <{transpose_b = true}> : (!lhs, !rhs) -> (!matmul_result)
    return %r : !matmul_result
  }
}

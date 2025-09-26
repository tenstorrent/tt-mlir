// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="use-tile-matmul=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

!ttype = tensor<128x96xf32>

!lhs = tensor<128x96xf32>
!rhs = tensor<96x64xf32>
!matmul_result = tensor<128x64xf32>

module {
  // CHECK-LABEL: func.func @test_block_matmul
  func.func @test_block_matmul(%lhs: !lhs, %rhs: !rhs, %out: !matmul_result) -> (!matmul_result) {
    // CHECK-NOT: ttir.matmul
    // CHECK-NOT: matmul_tiles
    // CHECK: mm_block_init
    // CHECK: mm_block_init_short
    // CHECK: matmul_block
    %r = "ttir.matmul"(%lhs, %rhs, %out) : (!lhs, !rhs, !matmul_result) -> (!matmul_result)
    return %r : !matmul_result
  }
}

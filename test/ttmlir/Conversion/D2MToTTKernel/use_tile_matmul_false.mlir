// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="use-tile-matmul=false,dst-allocation-strategy=legacy" %s | FileCheck %s --check-prefix=COMMON
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="use-tile-matmul=false,dst-allocation-strategy=graph-coloring-greedy" %s | FileCheck %s --check-prefix=COMMON
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="use-tile-matmul=false,dst-allocation-strategy=graph-coloring-cb" %s | FileCheck %s --check-prefix=COMMON

!ttype = tensor<128x96xf32>

!lhs = tensor<128x96xf32>
!rhs = tensor<96x64xf32>
!matmul_result = tensor<128x64xf32>

module {
  // COMMON-LABEL: func.func @test_block_matmul
  func.func @test_block_matmul(%lhs: !lhs, %rhs: !rhs, %out: !matmul_result) -> (!matmul_result) {
    // COMMON-NOT: ttir.matmul
    // COMMON-NOT: matmul_tiles
    // COMMON: mm_block_init
    // COMMON: mm_block_init_short
    // COMMON: matmul_block
    %r = "ttir.matmul"(%lhs, %rhs, %out) : (!lhs, !rhs, !matmul_result) -> (!matmul_result)
    return %r : !matmul_result
  }
}

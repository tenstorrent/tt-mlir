// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="use-tile-matmul=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

!ttype = tensor<128x96xf32>

!lhs = tensor<128x96xf32>
!rhs = tensor<96x64xf32>
!matmul_result = tensor<128x64xf32>

module {
  // CHECK-LABEL: func.func @test_block_matmul
  func.func @test_block_matmul(%lhs: !lhs, %rhs: !rhs) -> (!matmul_result) {
    // CHECK-NOT: ttir.matmul
    // CHECK-NOT: matmul_tiles
    // CHECK-DAG: emitc.verbatim "Noc noc0(0);"
    // CHECK-DAG: noc0.async_write_multicast
    // CHECK-DAG: emitc.verbatim "Noc noc1(1);"
    // CHECK-DAG: noc1.async_write_multicast
    // CHECK-NOT: noc_async_write_barrier
    // CHECK: compute_kernel_hw_startup
    // CHECK: matmul_block_init
    // CHECK: matmul_block
    %r = "ttir.matmul"(%lhs, %rhs) : (!lhs, !rhs) -> (!matmul_result)
    return %r : !matmul_result
  }
}

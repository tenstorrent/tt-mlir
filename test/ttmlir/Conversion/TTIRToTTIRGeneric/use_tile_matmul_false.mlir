// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttir-generic="use-tile-matmul=false" %s | FileCheck %s

!ttype = tensor<128x96xf32>

!lhs = tensor<128x96xf32>
!rhs = tensor<96x64xf32>
!matmul_result = tensor<128x64xf32>

module {
  // CHECK-LABEL: func @named_contractions
  func.func @named_contractions(%lhs: !lhs, %rhs: !rhs, %out: !matmul_result) -> (!matmul_result) {
    // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #parallel, #reduction]
    // CHECK-NOT: linalg.generic
    // CHECK: ttir.tile_matmul_block
    %r = "ttir.matmul"(%lhs, %rhs, %out) : (!lhs, !rhs, !matmul_result) -> (!matmul_result)
    return %r : !matmul_result
  }
}

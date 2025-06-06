// RUN: ttmlir-opt --tt-register-device --ttir-to-ttir-generic %s | FileCheck %s

!ttype = tensor<128x96xf32>

!lhs = tensor<128x96xf32>
!rhs = tensor<96x64xf32>
!matmul_result = tensor<128x64xf32>

module {

  // CHECK-LABEL: func @named_elementwise
  func.func @named_elementwise(%lhs: !ttype, %rhs: !ttype, %out: !ttype) -> (!ttype) {
    // named elementwise op, binary:
    // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: ttir.tile_add
    %0 = "ttir.add"(%lhs, %rhs, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: ttir.tile_exp
    %1 = "ttir.exp"(%0, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: ttir.tile_mul
    %2 = "ttir.multiply"(%0, %1, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: ttir.tile_log
    %3 = "ttir.log"(%2, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: ttir.tile_maximum
    %4 = "ttir.maximum"(%2, %3, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: ttir.tile_sin
    %5 = "ttir.sin"(%4, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: ttir.tile_sub
    %6 = "ttir.subtract"(%4, %5, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: ttir.tile_ceil
    %7 = "ttir.ceil"(%6, %out) : (!ttype, !ttype) -> !ttype
    return %7: !ttype
  }

  // CHECK-LABEL: func @named_reductions_R
  func.func @named_reductions_R(%arg: !ttype) -> (tensor<1x96xf32>) {
    %0 = ttir.empty() : tensor<1x96xf32>
    // CHECK: ttir.constant
    // CHECK: ttir.generic{{.+}}iterator_types = [#reduction, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["reduction", "parallel"]
    // CHECK: ttir.tile_reduce_sum{{.+}}ttir<reduce_dim R>
    %1 = "ttir.sum"(%arg, %0) <{dim_arg = [-2: i32], keep_dim = true}> : (!ttype, tensor<1x96xf32>) -> tensor<1x96xf32>
    return %1: tensor<1x96xf32>
  }

  // CHECK-LABEL: func @named_reductions_C
  func.func @named_reductions_C(%arg: !ttype) -> (tensor<128x1xf32>) {
    %0 = ttir.empty() : tensor<128x1xf32>
    // CHECK: ttir.constant
    // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #reduction]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "reduction"]
    // CHECK: ttir.tile_reduce_sum{{.+}}ttir<reduce_dim C>
    %1 = "ttir.sum"(%arg, %0) <{dim_arg = [-1: i32], keep_dim = true}> : (!ttype, tensor<128x1xf32>) -> tensor<128x1xf32>
    return %1 : tensor<128x1xf32>
  }

  // CHECK-LABEL: func @named_reductions_RC
  func.func @named_reductions_RC(%arg: !ttype) -> (tensor<1x1xf32>) {
    %0 = ttir.empty() : tensor<1x1xf32>
    // CHECK: ttir.constant
    // CHECK: ttir.generic{{.+}}iterator_types = [#reduction, #reduction]
    // CHECK: linalg.generic{{.+}}iterator_types = ["reduction", "reduction"]
    // CHECK: ttir.tile_reduce_sum{{.+}}ttir<reduce_dim RC>
    %1 = "ttir.sum"(%arg, %0) <{dim_arg = [-2: i32, -1: i32], keep_dim = true}> : (!ttype, tensor<1x1xf32>) -> tensor<1x1xf32>
    return %1: tensor<1x1xf32>
  }

  // CHECK-LABEL: func @named_contractions
  func.func @named_contractions(%lhs: !lhs, %rhs: !rhs, %out: !matmul_result) -> (!matmul_result) {
    // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #parallel, #reduction]
    // CHECK: linalg.generic
    // CHECK: ttir.tile_matmul
    %r = "ttir.matmul"(%lhs, %rhs, %out) : (!lhs, !rhs, !matmul_result) -> (!matmul_result)
    return %r : !matmul_result
  }
}

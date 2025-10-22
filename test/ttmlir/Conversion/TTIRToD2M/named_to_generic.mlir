// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m -o %t %s
// RUN: FileCheck %s --input-file=%t

!ttype = tensor<128x96xf32>

!lhs = tensor<128x96xf32>
!rhs = tensor<96x64xf32>
!matmul_result = tensor<128x64xf32>

module {

  // CHECK-LABEL: func @named_elementwise
  func.func @named_elementwise(%lhs: !ttype, %rhs: !ttype, %out: !ttype) -> (!ttype) {
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_add"(%{{.*}}, %{{.*}})
    %0 = "ttir.add"(%lhs, %rhs, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_exp
    %1 = "ttir.exp"(%0, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_mul"(%{{.*}}, %{{.*}})
    %2 = "ttir.multiply"(%0, %1, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_log
    %3 = "ttir.log"(%2, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_maximum
    %4 = "ttir.maximum"(%2, %3, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_sin
    %5 = "ttir.sin"(%4, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    %6 = "ttir.subtract"(%4, %5, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_ceil
    %7 = "ttir.ceil"(%6, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_rsqrt
    %8 = "ttir.rsqrt"(%7, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_pow
    %9 = "ttir.pow"(%7, %8, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_negative
    %10 = "ttir.neg"(%9, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_cos
    %11 = "ttir.cos"(%10, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_tan
    %12 = "ttir.tan"(%11, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_div
    %13 = "ttir.div"(%11, %12, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_abs
    %14 = "ttir.abs"(%13, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_sqrt
    %15 = "ttir.sqrt"(%14, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_recip
    %16 = "ttir.reciprocal"(%15, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_floor
    %17 = "ttir.floor"(%16, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_logical_not
    %18 = "ttir.logical_not"(%17, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_eqz
    %19 = "ttir.eq"(%lhs, %rhs, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_nez
    %20 = "ttir.ne"(%lhs, %rhs, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_ltz
    %21 = "ttir.lt"(%lhs, %rhs, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_lez
    %22 = "ttir.le"(%lhs, %rhs, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_gtz
    %23 = "ttir.gt"(%lhs, %rhs, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_gez
    %24 = "ttir.ge"(%lhs, %rhs, %out) : (!ttype, !ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_gelu
    %25= "ttir.gelu"(%24, %out) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_bitwise_not
    %26 = "ttir.bitwise_not"(%25, %out) : (!ttype, !ttype) -> !ttype
    return %26: !ttype
  }

  // CHECK-LABEL: func @named_reductions_R
  func.func @named_reductions_R(%arg: !ttype) -> (tensor<1x96xf32>) {
    %0 = ttir.empty() : tensor<1x96xf32>
    // CHECK: d2m.full
    // CHECK: d2m.generic{{.+}}iterator_types = [#reduction, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["reduction", "parallel"]
    // CHECK: d2m.tile_reduce_sum{{.+}}d2m<reduce_dim C>
    %1 = "ttir.sum"(%arg, %0) <{dim_arg = [-2: i32], keep_dim = true}> : (!ttype, tensor<1x96xf32>) -> tensor<1x96xf32>
    return %1: tensor<1x96xf32>
  }

  // CHECK-LABEL: func @named_reductions_C
  func.func @named_reductions_C(%arg: !ttype) -> (tensor<128x1xf32>) {
    %0 = ttir.empty() : tensor<128x1xf32>
    // CHECK: d2m.full
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #reduction]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "reduction"]
    // CHECK: d2m.tile_reduce_sum{{.+}}d2m<reduce_dim R>
    %1 = "ttir.sum"(%arg, %0) <{dim_arg = [-1: i32], keep_dim = true}> : (!ttype, tensor<128x1xf32>) -> tensor<128x1xf32>
    return %1 : tensor<128x1xf32>
  }

  // CHECK-LABEL: func @named_reductions_RC
  func.func @named_reductions_RC(%arg: !ttype) -> (tensor<1x1xf32>) {
    %0 = ttir.empty() : tensor<1x1xf32>
    // CHECK: d2m.full
    // CHECK: d2m.generic{{.+}}iterator_types = [#reduction, #reduction]
    // CHECK: linalg.generic{{.+}}iterator_types = ["reduction", "reduction"]
    // CHECK: d2m.tile_reduce_sum{{.+}}d2m<reduce_dim RC>
    %1 = "ttir.sum"(%arg, %0) <{dim_arg = [-2: i32, -1: i32], keep_dim = true}> : (!ttype, tensor<1x1xf32>) -> tensor<1x1xf32>
    return %1: tensor<1x1xf32>
  }

  // CHECK-LABEL: func @named_contractions
  func.func @named_contractions(%lhs: !lhs, %rhs: !rhs, %out: !matmul_result) -> (!matmul_result) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel, #reduction]
    // CHECK: linalg.generic
    // CHECK: d2m.tile_matmul
    %r = "ttir.matmul"(%lhs, %rhs, %out) : (!lhs, !rhs, !matmul_result) -> (!matmul_result)
    return %r : !matmul_result
  }
}

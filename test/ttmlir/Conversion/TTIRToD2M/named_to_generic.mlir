// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t

!ttype = tensor<128x96xf32>
!ttype_col = tensor<128x1xf32>
!ttype_row = tensor<1x96xf32>
!ttype_scalar = tensor<1x1xf32>

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
    %0 = "ttir.add"(%lhs, %rhs) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_exp
    %1 = "ttir.exp"(%0) : (!ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_mul"(%{{.*}}, %{{.*}})
    %2 = "ttir.multiply"(%0, %1) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_log
    %3 = "ttir.log"(%2) : (!ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_maximum
    %4 = "ttir.maximum"(%2, %3) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_sin
    %5 = "ttir.sin"(%4) : (!ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    %6 = "ttir.subtract"(%4, %5) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_ceil
    %7 = "ttir.ceil"(%6) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_rsqrt
    %8 = "ttir.rsqrt"(%7) : (!ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_pow
    %9 = "ttir.pow"(%7, %8) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_negative
    %10 = "ttir.neg"(%9) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_cos
    %11 = "ttir.cos"(%10) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_tan
    %12 = "ttir.tan"(%11) : (!ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_div
    %13 = "ttir.div"(%11, %12) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_abs
    %14 = "ttir.abs"(%13) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_sqrt
    %15 = "ttir.sqrt"(%14) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_recip
    %16 = "ttir.reciprocal"(%15) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_floor
    %17 = "ttir.floor"(%16) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_logical_not
    %18 = "ttir.logical_not"(%17) : (!ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_eqz
    %19 = "ttir.eq"(%lhs, %rhs) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_nez
    %20 = "ttir.ne"(%lhs, %rhs) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_ltz
    %21 = "ttir.lt"(%lhs, %rhs) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_lez
    %22 = "ttir.le"(%lhs, %rhs) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_gtz
    %23 = "ttir.gt"(%lhs, %rhs) : (!ttype, !ttype) -> !ttype
    // named elementwise op, binary comparison:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: "d2m.tile_sub"(%{{.*}}, %{{.*}})
    // CHECK: d2m.tile_gez
    %24 = "ttir.ge"(%lhs, %rhs) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_gelu
    %25 = "ttir.gelu"(%24) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_erf
    %26 = "ttir.erf"(%25) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_erfc
    %27 = "ttir.erfc"(%26) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_sign
    %28 = "ttir.sign"(%27) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_bitwise_not
    %29 = "ttir.bitwise_not"(%28) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_silu
    %30 = "ttir.silu"(%29) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_relu
    %31= "ttir.relu"(%30) : (!ttype) -> !ttype
    // named elementwise op, bitwise_and:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_bitwise_and
    %32 = "ttir.bitwise_and"(%26, %27) : (!ttype, !ttype) -> !ttype
    // named elementwise op, bitwise_or:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_bitwise_or
    %33 = "ttir.bitwise_or"(%26, %27) : (!ttype, !ttype) -> !ttype
    // named elementwise op, bitwise_xor:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_bitwise_xor
    %34 = "ttir.bitwise_xor"(%26, %27) : (!ttype, !ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_tanh
    %35 = "ttir.tanh"(%34) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_sigmoid
    %36 = "ttir.sigmoid"(%35) : (!ttype) -> !ttype
    // named elementwise op, unary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_hardsigmoid
    %37 = "ttir.hardsigmoid"(%36) : (!ttype) -> !ttype
    // named elementwise op, binary:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_minimum
    %38 = "ttir.minimum"(%37, %36) : (!ttype, !ttype) -> !ttype
    return %38: !ttype
  }

  // CHECK-LABEL: func @named_reductions_R
  func.func @named_reductions_R(%arg: !ttype) -> (tensor<1x96xf32>) {
    // CHECK: d2m.full
    // CHECK: d2m.generic{{.+}}iterator_types = [#reduction, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["reduction", "parallel"]
    // CHECK: d2m.tile_reduce_sum{{.+}}d2m<reduce_dim C>
    %1 = "ttir.sum"(%arg) <{dim_arg = [-2: i32], keep_dim = true}> : (!ttype) -> tensor<1x96xf32>
    return %1: tensor<1x96xf32>
  }

  // CHECK-LABEL: func @named_reductions_C
  func.func @named_reductions_C(%arg: !ttype) -> (tensor<128x1xf32>) {
    // CHECK: d2m.full
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #reduction]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "reduction"]
    // CHECK: d2m.tile_reduce_sum{{.+}}d2m<reduce_dim R>
    %1 = "ttir.sum"(%arg) <{dim_arg = [-1: i32], keep_dim = true}> : (!ttype) -> tensor<128x1xf32>
    return %1 : tensor<128x1xf32>
  }

  // CHECK-LABEL: func @named_reductions_RC
  func.func @named_reductions_RC(%arg: !ttype) -> (tensor<1x1xf32>) {
    // CHECK: d2m.full
    // CHECK: d2m.generic{{.+}}iterator_types = [#reduction, #reduction]
    // CHECK: linalg.generic{{.+}}iterator_types = ["reduction", "reduction"]
    // CHECK: d2m.tile_reduce_sum{{.+}}d2m<reduce_dim RC>
    %1 = "ttir.sum"(%arg) <{dim_arg = [-2: i32, -1: i32], keep_dim = true}> : (!ttype) -> tensor<1x1xf32>
    return %1: tensor<1x1xf32>
  }

  // CHECK-LABEL: func @named_contractions
  func.func @named_contractions(%lhs: !lhs, %rhs: !rhs) -> (!matmul_result) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel, #reduction]
    // CHECK: linalg.generic
    // CHECK: d2m.tile_matmul
    %r = "ttir.matmul"(%lhs, %rhs) : (!lhs, !rhs) -> (!matmul_result)
    return %r : !matmul_result
  }

  // CHECK-LABEL: func @implicit_bcast_2d_dual
  func.func @implicit_bcast_2d_dual(%in0: !ttype_col, %in1: !ttype_row) -> (!ttype) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type row>}>
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%in0, %in1) : (!ttype_col, !ttype_row) -> (!ttype)
    return %0 : !ttype
  }

  // CHECK-LABEL: func @implicit_bcast_2d_scalar
  func.func @implicit_bcast_2d_scalar(%in0: !ttype, %in1: !ttype_scalar) -> (!ttype) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type scalar>}>
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%in0, %in1) : (!ttype, !ttype_scalar) -> (!ttype)
    return %0 : !ttype
  }

  // CHECK-LABEL: func @named_ternary_where
  func.func @named_ternary_where(%cond: !ttype, %true_val: !ttype, %false_val: !ttype) -> (!ttype) {
    // named ternary op, where:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_where
    %0 = "ttir.where"(%cond, %true_val, %false_val) : (!ttype, !ttype, !ttype) -> !ttype
    return %0 : !ttype
  }

  // CHECK-LABEL: func @named_slice_static
  func.func @named_slice_static(%arg0: tensor<96x96xf32>) -> tensor<32x32xf32> {
    // CHECK-NOT: slice
    // CHECK: %[[DEVICE_TENSOR:.*]] = d2m.to_layout %arg0
    // CHECK: "d2m.stream_layout"(%[[DEVICE_TENSOR]], %{{.*}})
    // CHECK: d2m.generic
    %0 = "ttir.slice_static"(%arg0) <{begins = [1 : i32, 0 : i32], ends = [96 : i32, 64 : i32], step = [3 : i32, 2 : i32]}> : (tensor<96x96xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }

  // CHECK-LABEL: func @named_clamp_scalar
  func.func @named_clamp_scalar(%arg: !ttype) -> (!ttype) {
    // named clamp_scalar op, unary with scalar attributes:
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_clamp_scalar
    %0 = "ttir.clamp_scalar"(%arg) <{min = 2.000000e+00 : f32, max = 5.000000e+00 : f32}> : (!ttype) -> !ttype
    return %0 : !ttype
  }

  // CHECK-LABEL: func @named_clamp_tensor
  func.func @named_clamp_tensor(%input: !ttype, %min: !ttype, %max: !ttype) -> (!ttype) {
    // named clamp_tensor op, ternary:
    // clamp_tensor is decomposed into maximum(input, min) then minimum(result, max)
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_maximum
    // CHECK: d2m.tile_minimum
    %0 = "ttir.clamp_tensor"(%input, %min, %max) : (!ttype, !ttype, !ttype) -> !ttype
    return %0 : !ttype
  }

  // CHECK-LABEL: func @named_logical_and
  func.func @named_logical_and(%lhs: !ttype, %rhs: !ttype) -> (!ttype) {
    // logical_and is decomposed into: NEZ(a) * NEZ(b) - both must be non-zero
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_nez
    // CHECK: d2m.tile_nez
    // CHECK: "d2m.tile_mul"
    %0 = "ttir.logical_and"(%lhs, %rhs) : (!ttype, !ttype) -> !ttype
    return %0 : !ttype
  }

  // CHECK-LABEL: func @named_logical_or
  func.func @named_logical_or(%lhs: !ttype, %rhs: !ttype) -> (!ttype) {
    // logical_or is decomposed into: NEZ(NEZ(a) + NEZ(b)) - at least one must be non-zero
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_nez
    // CHECK: d2m.tile_nez
    // CHECK: "d2m.tile_add"
    // CHECK: d2m.tile_nez
    %0 = "ttir.logical_or"(%lhs, %rhs) : (!ttype, !ttype) -> !ttype
    return %0 : !ttype
  }

  // CHECK-LABEL: func @named_logical_xor
  func.func @named_logical_xor(%lhs: !ttype, %rhs: !ttype) -> (!ttype) {
    // logical_xor is decomposed into: NEZ(NEZ(a) - NEZ(b)) - exactly one must be non-zero
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
    // CHECK: d2m.tile_nez
    // CHECK: d2m.tile_nez
    // CHECK: "d2m.tile_sub"
    // CHECK: d2m.tile_nez
    %0 = "ttir.logical_xor"(%lhs, %rhs) : (!ttype, !ttype) -> !ttype
    return %0 : !ttype
  }
}

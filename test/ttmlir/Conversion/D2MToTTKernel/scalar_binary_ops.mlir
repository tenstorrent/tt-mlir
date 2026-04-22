// RUN: ttmlir-opt --split-input-file --ttir-to-ttmetal-fe-pipeline --ttir-to-ttmetal-me-pipeline --convert-d2m-to-ttkernel %s | FileCheck %s

// Scalar add: f32 uses the float path (add_unary_tile).

!ttype_f32 = tensor<32x32xf32>
// CHECK-LABEL: func.func @test_add_scalar_f32
func.func @test_add_scalar_f32(%in: !ttype_f32) -> (!ttype_f32) {
  // CHECK: ttkernel.binop_with_scalar_tile_init
  // CHECK: ttkernel.add_unary_tile(
  // CHECK-NOT: ttkernel.add_unary_tile_int32
  %cst = "ttir.constant"() {value = dense<2.500000e+00> : tensor<32x32xf32>} : () -> !ttype_f32
  %0 = "ttir.add"(%in, %cst) : (!ttype_f32, !ttype_f32) -> !ttype_f32
  return %0 : !ttype_f32
}

// -----

// Scalar add: bf16 uses the float path (add_unary_tile).

!ttype_bf16 = tensor<32x32xbf16>
// CHECK-LABEL: func.func @test_add_scalar_bf16
func.func @test_add_scalar_bf16(%in: !ttype_bf16) -> (!ttype_bf16) {
  // CHECK: ttkernel.binop_with_scalar_tile_init
  // CHECK: ttkernel.add_unary_tile(
  // CHECK-NOT: ttkernel.add_unary_tile_int32
  %cst = "ttir.constant"() {value = dense<2.500000e+00> : tensor<32x32xbf16>} : () -> !ttype_bf16
  %0 = "ttir.add"(%in, %cst) : (!ttype_bf16, !ttype_bf16) -> !ttype_bf16
  return %0 : !ttype_bf16
}

// -----

// Scalar add: i32 uses the int32 path (add_unary_tile_int32).

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_add_scalar_i32
func.func @test_add_scalar_i32(%in: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.binop_with_scalar_tile_init
  // CHECK: ttkernel.add_unary_tile_int32(
  %cst = "ttir.constant"() {value = dense<5> : tensor<32x32xsi32>} : () -> !ttype_i32
  %0 = "ttir.add"(%in, %cst) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

// Scalar subtract: f32 uses the float path (sub_unary_tile).

!ttype_f32 = tensor<32x32xf32>
// CHECK-LABEL: func.func @test_subtract_scalar_f32
func.func @test_subtract_scalar_f32(%in: !ttype_f32) -> (!ttype_f32) {
  // CHECK: ttkernel.binop_with_scalar_tile_init
  // CHECK: ttkernel.sub_unary_tile(
  // CHECK-NOT: ttkernel.sub_unary_tile_int32
  %cst = "ttir.constant"() {value = dense<1.500000e+00> : tensor<32x32xf32>} : () -> !ttype_f32
  %0 = "ttir.subtract"(%in, %cst) : (!ttype_f32, !ttype_f32) -> !ttype_f32
  return %0 : !ttype_f32
}

// -----

// Scalar subtract: bf16 uses the float path (sub_unary_tile).

!ttype_bf16 = tensor<32x32xbf16>
// CHECK-LABEL: func.func @test_subtract_scalar_bf16
func.func @test_subtract_scalar_bf16(%in: !ttype_bf16) -> (!ttype_bf16) {
  // CHECK: ttkernel.binop_with_scalar_tile_init
  // CHECK: ttkernel.sub_unary_tile(
  // CHECK-NOT: ttkernel.sub_unary_tile_int32
  %cst = "ttir.constant"() {value = dense<1.500000e+00> : tensor<32x32xbf16>} : () -> !ttype_bf16
  %0 = "ttir.subtract"(%in, %cst) : (!ttype_bf16, !ttype_bf16) -> !ttype_bf16
  return %0 : !ttype_bf16
}

// -----

// Scalar subtract: i32 uses the int32 path (sub_unary_tile_int32).

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_subtract_scalar_i32
func.func @test_subtract_scalar_i32(%in: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.binop_with_scalar_tile_init
  // CHECK: ttkernel.sub_unary_tile_int32(
  %cst = "ttir.constant"() {value = dense<3> : tensor<32x32xsi32>} : () -> !ttype_i32
  %0 = "ttir.subtract"(%in, %cst) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

// Scalar multiply: f32 uses mul_unary_tile.

!ttype_f32 = tensor<32x32xf32>
// CHECK-LABEL: func.func @test_multiply_scalar_f32
func.func @test_multiply_scalar_f32(%in: !ttype_f32) -> (!ttype_f32) {
  // CHECK: ttkernel.binop_with_scalar_tile_init
  // CHECK: ttkernel.mul_unary_tile(
  %cst = "ttir.constant"() {value = dense<3.000000e+00> : tensor<32x32xf32>} : () -> !ttype_f32
  %0 = "ttir.multiply"(%in, %cst) : (!ttype_f32, !ttype_f32) -> !ttype_f32
  return %0 : !ttype_f32
}

// -----

// Scalar multiply: bf16 uses mul_unary_tile.

!ttype_bf16 = tensor<32x32xbf16>
// CHECK-LABEL: func.func @test_multiply_scalar_bf16
func.func @test_multiply_scalar_bf16(%in: !ttype_bf16) -> (!ttype_bf16) {
  // CHECK: ttkernel.binop_with_scalar_tile_init
  // CHECK: ttkernel.mul_unary_tile(
  %cst = "ttir.constant"() {value = dense<3.000000e+00> : tensor<32x32xbf16>} : () -> !ttype_bf16
  %0 = "ttir.multiply"(%in, %cst) : (!ttype_bf16, !ttype_bf16) -> !ttype_bf16
  return %0 : !ttype_bf16
}

// -----

// Scalar div: f32 uses div_unary_tile.

!ttype_f32 = tensor<32x32xf32>
// CHECK-LABEL: func.func @test_div_scalar_f32
func.func @test_div_scalar_f32(%in: !ttype_f32) -> (!ttype_f32) {
  // CHECK: ttkernel.div_unary_tile(
  %cst = "ttir.constant"() {value = dense<3.000000e+00> : tensor<32x32xf32>} : () -> !ttype_f32
  %0 = "ttir.div"(%in, %cst) : (!ttype_f32, !ttype_f32) -> !ttype_f32
  return %0 : !ttype_f32
}

// -----

// Scalar div: bf16 uses div_unary_tile.

!ttype_bf16 = tensor<32x32xbf16>
// CHECK-LABEL: func.func @test_div_scalar_bf16
func.func @test_div_scalar_bf16(%in: !ttype_bf16) -> (!ttype_bf16) {
  // CHECK: ttkernel.div_unary_tile(
  %cst = "ttir.constant"() {value = dense<3.000000e+00> : tensor<32x32xbf16>} : () -> !ttype_bf16
  %0 = "ttir.div"(%in, %cst) : (!ttype_bf16, !ttype_bf16) -> !ttype_bf16
  return %0 : !ttype_bf16
}

// -----

// Scalar pow: f32 uses power_tile_init + pow_unary_tile.

!ttype_f32 = tensor<32x32xf32>
// CHECK-LABEL: func.func @test_pow_scalar_f32
func.func @test_pow_scalar_f32(%in: !ttype_f32) -> (!ttype_f32) {
  // CHECK: ttkernel.power_tile_init
  // CHECK: ttkernel.power_tile(
  %cst = "ttir.constant"() {value = dense<2.000000e+00> : tensor<32x32xf32>} : () -> !ttype_f32
  %0 = "ttir.pow"(%in, %cst) : (!ttype_f32, !ttype_f32) -> !ttype_f32
  return %0 : !ttype_f32
}

// -----

// Scalar pow: bf16 uses power_tile_init + pow_unary_tile.

!ttype_bf16 = tensor<32x32xbf16>
// CHECK-LABEL: func.func @test_pow_scalar_bf16
func.func @test_pow_scalar_bf16(%in: !ttype_bf16) -> (!ttype_bf16) {
  // CHECK: ttkernel.power_tile_init
  // CHECK: ttkernel.power_tile(
  %cst = "ttir.constant"() {value = dense<2.000000e+00> : tensor<32x32xbf16>} : () -> !ttype_bf16
  %0 = "ttir.pow"(%in, %cst) : (!ttype_bf16, !ttype_bf16) -> !ttype_bf16
  return %0 : !ttype_bf16
}

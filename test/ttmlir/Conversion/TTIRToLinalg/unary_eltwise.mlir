// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @test_abs
func.func @test_abs(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.abs
  %0 = "ttir.abs"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_atan
func.func @test_atan(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.generic
  // CHECK: math.atan
  %0 = "ttir.atan"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_bitwise_not
func.func @test_bitwise_not(%arg0: tensor<64x128xi32>) -> tensor<64x128xi32> {
  // CHECK: tosa.bitwise_not
  %0 = "ttir.bitwise_not"(%arg0) : (tensor<64x128xi32>) -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}

// CHECK-LABEL: func.func @test_cbrt
func.func @test_cbrt(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.generic
  // CHECK: math.cbrt
  %0 = "ttir.cbrt"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_ceil
func.func @test_ceil(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.ceil
  %0 = "ttir.ceil"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_clamp_scalar
func.func @test_clamp_scalar(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.clamp
  %0 = "ttir.clamp_scalar"(%arg0) <{min = 2.000000e+00 : f32, max = 3.000000e+00 : f32}> : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_cos
func.func @test_cos(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.cos
  %0 = "ttir.cos"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_erf
func.func @test_erf(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.erf
  %0 = "ttir.erf"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_erfc
func.func @test_erfc(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.generic
  // CHECK: math.erfc
  %0 = "ttir.erfc"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_exp
func.func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: tosa.exp
  %0 = "ttir.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// CHECK-LABEL: func.func @test_expm1
func.func @test_expm1(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.generic
  // CHECK: math.expm1
  %0 = "ttir.expm1"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_floor
func.func @test_floor(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.floor
  %0 = "ttir.floor"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_gelu
func.func @test_gelu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.mul
  // CHECK: tosa.erf
  // CHECK: tosa.add
  // CHECK: tosa.mul
  // CHECK: tosa.mul
  %0 = "ttir.gelu"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_hardsigmoid
func.func @test_hardsigmoid(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.add
  // CHECK: tosa.mul
  // CHECK: tosa.clamp
  %0 = "ttir.hardsigmoid"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_isfinite
func.func @test_isfinite(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.subf
  // CHECK: arith.cmpf oeq
  // CHECK: arith.uitofp
  %0 = "ttir.isfinite"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_leaky_relu
func.func @test_leaky_relu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.mul
  // CHECK: tosa.greater
  // CHECK: tosa.select
  %0 = "ttir.leaky_relu"(%arg0) {parameter = 0.01 : f32} : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_log
func.func @test_log(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.log
  %0 = "ttir.log"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_log1p
func.func @test_log1p(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.generic
  // CHECK: math.log1p
  %0 = "ttir.log1p"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_logical_not
func.func @test_logical_not(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.logical_not
  // CHECK: tosa.cast
  %0 = "ttir.logical_not"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_mish
func.func @test_mish(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.abs
  // CHECK: tosa.negate
  // CHECK: tosa.exp
  // CHECK: tosa.add
  // CHECK: tosa.log
  // CHECK: tosa.maximum
  // CHECK: tosa.add
  // CHECK: tosa.tanh
  // CHECK: tosa.mul
  %0 = "ttir.mish"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_neg
func.func @test_neg(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.negate
  %0 = "ttir.neg"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_reciprocal
func.func @test_reciprocal(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.reciprocal
  %0 = "ttir.reciprocal"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_relu
func.func @test_relu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: arith.constant
  // CHECK: linalg.max
  %0 = "ttir.relu"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_relu_bf16
func.func @test_relu_bf16(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  // CHECK: arith.constant
  // CHECK: linalg.max
  %0 = "ttir.relu"(%arg0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %0 : tensor<64x128xbf16>
}

// CHECK-LABEL: func.func @test_relu6
func.func @test_relu6(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.clamp
  %0 = "ttir.relu6"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_rsqrt
func.func @test_rsqrt(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.generic
  // CHECK: math.rsqrt
  %0 = "ttir.rsqrt"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_sigmoid
func.func @test_sigmoid(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.sigmoid
  %0 = "ttir.sigmoid"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_sign
func.func @test_sign(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.greater
  // CHECK: tosa.equal
  // CHECK: tosa.select
  // CHECK: tosa.select
  %0 = "ttir.sign"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_silu
func.func @test_silu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.sigmoid
  // CHECK: tosa.mul
  %0 = "ttir.silu"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_sin
func.func @test_sin(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.sin
  %0 = "ttir.sin"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_sqrt
func.func @test_sqrt(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.sqrt
  %0 = "ttir.sqrt"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_tan
func.func @test_tan(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.generic
  // CHECK: math.tan
  %0 = "ttir.tan"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_tanh
func.func @test_tanh(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.tanh
  %0 = "ttir.tanh"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_typecast
func.func @test_typecast(%arg0: tensor<64x128xf32>) -> tensor<64x128xi32> {
  // CHECK: tosa.cast
  %0 = "ttir.typecast"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}

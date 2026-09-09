// RUN: ttmlir-opt --canonicalize --ttir-element-type-normalization -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func public @constant_i1() -> tensor<32x32xi1> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xbf16>
  %0 = "ttir.full"() <{fill_value = 1 : i32, shape = array<i32: 32, 32>}> : () -> tensor<32x32xi1>
  return %0 : tensor<32x32xi1>
}

func.func public @constant_i8() -> tensor<32x32xi8> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xui8>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xi8>}> : () -> tensor<32x32xi8>
  return %0 : tensor<32x32xi8>
}

func.func public @constant_i16() -> tensor<32x32xi16> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xui16>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xi16>}> : () -> tensor<32x32xi16>
  return %0 : tensor<32x32xi16>
}

func.func @constant_i32() -> tensor<32x32xi32> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xsi32>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xi32>}> : () -> tensor<32x32xi32>
  return %0 : tensor<32x32xi32>
}

func.func @constant_i64() -> tensor<32x32xi64> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xsi32>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xi64>}> : () -> tensor<32x32xi64>
  return %0 : tensor<32x32xi64>
}

func.func @constant_ui1() -> tensor<32x32xui1> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xbf16>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xui1>}> : () -> tensor<32x32xui1>
  return %0 : tensor<32x32xui1>
}

func.func @constant_ui8() -> tensor<32x32xui8> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xui8>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xui8>}> : () -> tensor<32x32xui8>
  return %0 : tensor<32x32xui8>
}

func.func @constant_ui16() -> tensor<32x32xui16> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xui16>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xui16>}> : () -> tensor<32x32xui16>
  return %0 : tensor<32x32xui16>
}

func.func @constant_ui32() -> tensor<32x32xui32> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xui32>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xui32>}> : () -> tensor<32x32xui32>
  return %0 : tensor<32x32xui32>
}

func.func @constant_ui64() -> tensor<32x32xui64> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xui32>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xui64>}> : () -> tensor<32x32xui64>
  return %0 : tensor<32x32xui64>
}

func.func @constant_f16() -> tensor<32x32xf16> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1.000000e+00 : f32
  // CHECK-SAME: -> tensor<32x32xbf16>
  %0 = "ttir.constant"() <{value = dense<1.0> : tensor<32x32xf16>}> : () -> tensor<32x32xf16>
  return %0 : tensor<32x32xf16>
}

func.func @constant_f32() -> tensor<32x32xf32> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1.000000e+00 : f32
  // CHECK-SAME: -> tensor<32x32xf32>
  %0 = "ttir.constant"() <{value = dense<1.0> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

func.func @constant_f64() -> tensor<32x32xf64> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1.000000e+00 : f32
  // CHECK-SAME: -> tensor<32x32xf32>
  %0 = "ttir.constant"() <{value = dense<1.0> : tensor<32x32xf64>}> : () -> tensor<32x32xf64>
  return %0 : tensor<32x32xf64>
}

func.func @constant_si8() -> tensor<32x32xsi8> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xui8>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xsi8>}> : () -> tensor<32x32xsi8>
  return %0 : tensor<32x32xsi8>
}

func.func @constant_si16() -> tensor<32x32xsi16> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xui16>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xsi16>}> : () -> tensor<32x32xsi16>
  return %0 : tensor<32x32xsi16>
}

func.func @constant_si32() -> tensor<32x32xsi32> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xsi32>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xsi32>}> : () -> tensor<32x32xsi32>
  return %0 : tensor<32x32xsi32>
}

func.func @constant_si64() -> tensor<32x32xsi64> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1 : i32
  // CHECK-SAME: -> tensor<32x32xsi32>
  %0 = "ttir.constant"() <{value = dense<1> : tensor<32x32xsi64>}> : () -> tensor<32x32xsi64>
  return %0 : tensor<32x32xsi64>
}

func.func @constant_bf16() -> tensor<32x32xbf16> {
  // CHECK: "ttir.full"
  // CHECK-SAME: value = 1.000000e+00 : f32
  // CHECK-SAME: -> tensor<32x32xbf16>
  %0 = "ttir.constant"() <{value = dense<1.0> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

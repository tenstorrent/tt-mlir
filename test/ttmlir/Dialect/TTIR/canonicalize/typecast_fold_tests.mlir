// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @typecast_const_f32_to_si8() -> tensor<3xsi8> {
    %0 = "ttir.constant"() {
      value = dense<[1.700000e+00, -2.300000e+00, 3.000000e+00]> : tensor<3xf32>
    } : () -> tensor<3xf32>

    // CHECK-LABEL: func.func @typecast_const_f32_to_si8
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1, -2, 3]> : tensor<3xsi8>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<3xf32>) -> tensor<3xsi8>
    return %1 : tensor<3xsi8>
  }

  func.func @no_fold_f32_to_si8() -> tensor<3xi8> {
    %0 = "ttir.full"() {
      fill_value = 128.000000e+00 : f32,
      shape = array<i32: 3>
    } : () -> tensor<3xf32>

    // CHECK-LABEL: func.func @no_fold_f32_to_si8
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 1.280000e+02 : f32
    // CHECK: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<3xf32>) -> tensor<3xi8>
    return %1 : tensor<3xi8>
  }

  func.func @typecast_const_f32_to_ui8() -> tensor<3xui8> {
    %0 = "ttir.constant"() {
      value = dense<[1.700000e+00, 222.300000e+00, 3.000000e+00]> : tensor<3xf32>
    } : () -> tensor<3xf32>

    // CHECK-LABEL: func.func @typecast_const_f32_to_ui8
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1, 222, 3]> : tensor<3xui8>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<3xf32>) -> tensor<3xui8>
    return %1 : tensor<3xui8>
  }

  func.func @typecast_const_si32_to_f32() -> tensor<3xf32> {
    %0 = "ttir.constant"() {
      value = dense<[1, -2, 3]> : tensor<3xsi32>
    } : () -> tensor<3xsi32>

    // CHECK-LABEL: func.func @typecast_const_si32_to_f32
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1.000000e+00, -2.000000e+00, 3.000000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<3xsi32>) -> tensor<3xf32>
    return %1 : tensor<3xf32>
  }

  func.func @typecast_const_f32_to_bf16() -> tensor<4xbf16> {
    %0 = "ttir.constant"() {
      value = dense<[1.000000e+00, 2.000000e+00, 3.141593e+00, -4.000000e+00]> : tensor<4xf32>
    } : () -> tensor<4xf32>

    // CHECK-LABEL: func.func @typecast_const_f32_to_bf16
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1.000000e+00, 2.000000e+00, 3.14{{[0-9]*(e\+00)?}}, -4.000000e+00]> : tensor<4xbf16>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xf32>) -> tensor<4xbf16>
    return %1 : tensor<4xbf16>
  }

  func.func @typecast_const_f16_to_f32() -> tensor<4xf32> {
    %0 = "ttir.constant"() {
      value = dense<[1.000000e+00, 2.000000e+00, 3.141593e+00, -4.000000e+00]> : tensor<4xf16>
    } : () -> tensor<4xf16>

    // CHECK-LABEL: func.func @typecast_const_f16_to_f32
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1.000000e+00, 2.000000e+00, 3.14{{[0-9]*(e\+00)?}}, -4.000000e+00]> : tensor<4xf32>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xf16>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  func.func @typecast_const_ui16_to_ui8() -> tensor<4xui8> {
    %0 = "ttir.constant"() {
      value = dense<[255, 0, 1, 42]> : tensor<4xui16>
    } : () -> tensor<4xui16>
    // CHECK-LABEL: func.func @typecast_const_ui16_to_ui8
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[255, 0, 1, 42]> : tensor<4xui8>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xui16>) -> tensor<4xui8>
    return %1 : tensor<4xui8>
  }

  func.func @typecast_const_ui8_to_ui16() -> tensor<4xui16> {
    %0 = "ttir.constant"() {
      value = dense<[255, 0, 1, 42]> : tensor<4xui8>
    } : () -> tensor<4xui8>
    // CHECK-LABEL: func.func @typecast_const_ui8_to_ui16
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[255, 0, 1, 42]> : tensor<4xui16>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xui8>) -> tensor<4xui16>
    return %1 : tensor<4xui16>
  }

  func.func @typecast_const_si32_to_si8() -> tensor<4xsi8> {
    %0 = "ttir.constant"() {
      value = dense<[127, -128, -1, 0]> : tensor<4xsi32>
    } : () -> tensor<4xsi32>

    // CHECK-LABEL: func.func @typecast_const_si32_to_si8
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[127, -128, -1, 0]> : tensor<4xsi8>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xsi32>) -> tensor<4xsi8>
    return %1 : tensor<4xsi8>
  }

  func.func @no_fold_si32_to_si8() -> tensor<4xsi8> {
    %0 = "ttir.constant"() {
      value = dense<[128, -129, -1, 0]> : tensor<4xsi32>
    } : () -> tensor<4xsi32>

    // CHECK-LABEL: func.func @no_fold_si32_to_si8
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[128, -129, -1, 0]> : tensor<4xsi32>
    // CHECK: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xsi32>) -> tensor<4xsi8>
    return %1 : tensor<4xsi8>
  }

  func.func @typecast_const_si8_to_ui8() -> tensor<4xui8> {
    %0 = "ttir.constant"() {
      value = dense<[127, 0, 1, 2]> : tensor<4xsi8>
    } : () -> tensor<4xsi8>

    // CHECK-LABEL: func.func @typecast_const_si8_to_ui8
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[127, 0, 1, 2]> : tensor<4xui8>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xsi8>) -> tensor<4xui8>
    return %1 : tensor<4xui8>
  }

  func.func @no_fold_si8_to_ui8() -> tensor<4xui8> {
    %0 = "ttir.constant"() {
      value = dense<[-1, 0, 1, 2]> : tensor<4xsi8>
    } : () -> tensor<4xsi8>

    // CHECK-LABEL: func.func @no_fold_si8_to_ui8
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[-1, 0, 1, 2]> : tensor<4xsi8>
    // CHECK: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xsi8>) -> tensor<4xui8>
    return %1 : tensor<4xui8>
  }

  func.func @typecast_const_ui8_to_si8() -> tensor<4xsi8> {
    %0 = "ttir.constant"() {
      value = dense<[127, 0, 1, 2]> : tensor<4xui8>
    } : () -> tensor<4xui8>

    // CHECK-LABEL: func.func @typecast_const_ui8_to_si8
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[127, 0, 1, 2]> : tensor<4xsi8>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xui8>) -> tensor<4xsi8>
    return %1 : tensor<4xsi8>
  }

  func.func @no_fold_ui8_to_si8() -> tensor<4xsi8> {
    %0 = "ttir.constant"() {
      value = dense<[128, 0, 1, 2]> : tensor<4xui8>
    } : () -> tensor<4xui8>

    // CHECK-LABEL: func.func @no_fold_ui8_to_si8
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[128, 0, 1, 2]> : tensor<4xui8>
    // CHECK: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xui8>) -> tensor<4xsi8>
    return %1 : tensor<4xsi8>
  }

  func.func @typecast_const_ui8_to_si16() -> tensor<4xsi16> {
    %0 = "ttir.full"() {
      fill_value = 255 : i32,
      shape = array<i32: 4>
    } : () -> tensor<4xui8>

    // CHECK-LABEL: func.func @typecast_const_ui8_to_si16
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 255 : i32
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xui8>) -> tensor<4xsi16>
    return %1 : tensor<4xsi16>
  }

  func.func @typecast_const_si16_to_ui8() -> tensor<4xui8> {
    %0 = "ttir.full"() {
      fill_value = 255 : i32,
      shape = array<i32: 4>
    } : () -> tensor<4xsi16>

    // CHECK-LABEL: func.func @typecast_const_si16_to_ui8
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 255 : i32
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xsi16>) -> tensor<4xui8>
    return %1 : tensor<4xui8>
  }

  func.func @typecast_const_i32_to_i8() -> tensor<4xi8> {
    %0 = "ttir.full"() {
      fill_value = 127 : i32,
      shape = array<i32: 4>
    } : () -> tensor<4xi32>

    // CHECK-LABEL: func.func @typecast_const_i32_to_i8
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 127 : i32
    // CHECK-SAME: -> tensor<4xi8>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xi32>) -> tensor<4xi8>
    return %1 : tensor<4xi8>
  }

  func.func @typecast_const_chain() -> tensor<4xbf16> {
    %0 = "ttir.constant"() {
      value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf16>
    } : () -> tensor<4xf16>
    // CHECK-LABEL: func.func @typecast_const_chain
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xbf16>
    // CHECK-NOT: "ttir.typecast"
    %1 = "ttir.typecast"(%0) : (tensor<4xf16>) -> tensor<4xf32>
    %2 = "ttir.typecast"(%1) : (tensor<4xf32>) -> tensor<4xbf16>
    return %2 : tensor<4xbf16>
  }
}

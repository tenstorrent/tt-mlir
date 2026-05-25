// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @clamp_tensor_int() -> tensor<3xsi32> {
    // CHECK-LABEL: @clamp_tensor_int
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1, 3, 3]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.clamp_tensor"
    %0 = "ttir.constant"() {value = dense<[-1, 5, 3]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[1, 2, -2]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.constant"() {value = dense<[2, 3, 4]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %3 = "ttir.clamp_tensor"(%0, %1, %2) : (tensor<3xsi32>, tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %3 : tensor<3xsi32>
  }

  func.func @clamp_tensor_f32() -> tensor<3xf32> {
    // CHECK-LABEL: @clamp_tensor_f32
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+00, 3.000000e+00, 3.000000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.clamp_tensor"
    %0 = "ttir.constant"() {value = dense<[-1.0, 5.0, 3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[1.0, 2.0, -2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %3 = "ttir.clamp_tensor"(%0, %1, %2) : (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %3 : tensor<3xf32>
  }

  func.func @clamp_tensor_bf16() -> tensor<3xbf16> {
    // CHECK-LABEL: @clamp_tensor_bf16
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+00, 3.000000e+00, 3.000000e+00]> : tensor<3xbf16>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.clamp_tensor"
    %0 = "ttir.constant"() {value = dense<[-1.0, 5.0, 3.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %1 = "ttir.constant"() {value = dense<[1.0, 2.0, -2.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %2 = "ttir.constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %3 = "ttir.clamp_tensor"(%0, %1, %2) : (tensor<3xbf16>, tensor<3xbf16>, tensor<3xbf16>) -> tensor<3xbf16>
    return %3 : tensor<3xbf16>
  }

  func.func @clamp_tensor_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @clamp_tensor_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1, 3, 3]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.clamp_tensor"
    %0 = "ttir.constant"() {value = dense<[0, 5, 3]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[1, 2, 0]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.constant"() {value = dense<[2, 3, 250]> : tensor<3xui8>} : () -> tensor<3xui8>
    %3 = "ttir.clamp_tensor"(%0, %1, %2) : (tensor<3xui8>, tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %3 : tensor<3xui8>
  }

  func.func @clamp_tensor_full() -> tensor<3xf32> {
    // CHECK-LABEL: @clamp_tensor_full
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 3.000000e+00 : f32
    // CHECK-NOT: "ttir.clamp"
    %0 = "ttir.full"() <{shape = array<i32: 3>, fill_value = 5.000000e+00 : f32}> : () -> tensor<3xf32>
    %1 = "ttir.ones"() <{shape = array<i32: 3>}> : () -> tensor<3xf32>
    %2 = "ttir.full"() <{shape = array<i32: 3>, fill_value = 3.000000e+00 : f32}> : () -> tensor<3xf32>
    %3 = "ttir.clamp_tensor"(%0, %1, %2) : (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %3 : tensor<3xf32>
  }

  func.func @where_int() -> tensor<3xsi32> {
    // CHECK-LABEL: @where_int
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}100, 20, 300]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.where"
    %0 = "ttir.constant"() {value = dense<[0, 1, 0]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[10, 20, 30]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.constant"() {value = dense<[100, 200, 300]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %3 = "ttir.where"(%0, %1, %2) : (tensor<3xsi32>, tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %3 : tensor<3xsi32>
  }

  func.func @where_f32() -> tensor<3xf32> {
    // CHECK-LABEL: @where_f32
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+02, 2.000000e+01, 3.000000e+02]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.where"
    %0 = "ttir.constant"() {value = dense<[0.0, 1.0, 0.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[10.0, 20.0, 30.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.constant"() {value = dense<[100.0, 200.0, 300.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %3 = "ttir.where"(%0, %1, %2) : (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %3 : tensor<3xf32>
  }

  func.func @where_f32_condition_i1() -> tensor<3xf32> {
    // CHECK-LABEL: @where_f32_condition_i1
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+02, 2.000000e+01, 3.000000e+02]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.where"
    %0 = "ttir.constant"() {value = dense<[0, 1, 0]> : tensor<3xi1>} : () -> tensor<3xi1>
    %1 = "ttir.constant"() {value = dense<[10.0, 20.0, 30.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.constant"() {value = dense<[100.0, 200.0, 300.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %3 = "ttir.where"(%0, %1, %2) : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %3 : tensor<3xf32>
  }

  func.func @where_bf16() -> tensor<3xbf16> {
    // CHECK-LABEL: @where_bf16
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+02, 2.000000e+01, 3.000000e+02]> : tensor<3xbf16>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.where"
    %0 = "ttir.constant"() {value = dense<[0.0, 1.0, 0.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %1 = "ttir.constant"() {value = dense<[10.0, 20.0, 30.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %2 = "ttir.constant"() {value = dense<[100.0, 200.0, 300.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %3 = "ttir.where"(%0, %1, %2) : (tensor<3xbf16>, tensor<3xbf16>, tensor<3xbf16>) -> tensor<3xbf16>
    return %3 : tensor<3xbf16>
  }

  func.func @where_full() -> tensor<3xf32> {
    // CHECK-LABEL: @where_full
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 1.000000e+01 : f32
    // CHECK-NOT: "ttir.where"
    %0 = "ttir.ones"() <{shape = array<i32: 3>}> : () -> tensor<3xf32>
    %1 = "ttir.full"() <{shape = array<i32: 3>, fill_value = 1.000000e+01 : f32}> : () -> tensor<3xf32>
    %2 = "ttir.full"() <{shape = array<i32: 3>, fill_value = 1.000000e+02 : f32}> : () -> tensor<3xf32>
    %3 = "ttir.where"(%0, %1, %2) : (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %3 : tensor<3xf32>
  }

  func.func @where_broadcastable() -> tensor<2x2xsi32> {
    // CHECK-LABEL: @where_broadcastable
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[\[}}10, 200], {{\[}}20, 200]]> : tensor<2x2xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.where"
    %0 = "ttir.constant"() <{value = dense<[1, 0]> : tensor<2xsi32>}> : () -> tensor<2xsi32>
    %1 = "ttir.constant"() <{value = dense<[[10], [20]]> : tensor<2x1xsi32>}> : () -> tensor<2x1xsi32>
    %2 = "ttir.constant"() <{value = dense<[100, 200]> : tensor<2xsi32>}> : () -> tensor<2xsi32>
    %3 = "ttir.where"(%0, %1, %2) : (tensor<2xsi32>, tensor<2x1xsi32>, tensor<2xsi32>) -> tensor<2x2xsi32>
    return %3 : tensor<2x2xsi32>
  }

  func.func @where_broadcastable_zeros() -> tensor<3xf32> {
    // CHECK-LABEL: @where_broadcastable_zeros
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+02, 2.000000e+02, 3.000000e+02]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.where"
    %0 = "ttir.zeros"() <{shape = array<i32: 1>}> : () -> tensor<1xf32>
    %1 = "ttir.constant"() {value = dense<[10.0, 20.0, 30.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.constant"() {value = dense<[100.0, 200.0, 300.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %3 = "ttir.where"(%0, %1, %2) : (tensor<1xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %3 : tensor<3xf32>
  }

  func.func @where_broadcastable_no_fold() -> tensor<2x2xsi32> {
    // CHECK-LABEL: @where_broadcastable_no_fold
    // CHECK: "ttir.zeros"
    // CHECK: "ttir.constant"
    // CHECK: "ttir.constant"
    // CHECK: "ttir.where"
    %0 = "ttir.zeros"() <{shape = array<i32: 2>}> : () -> tensor<2xsi32>
    %1 = "ttir.constant"() <{value = dense<[[10], [20]]> : tensor<2x1xsi32>}> : () -> tensor<2x1xsi32>
    %2 = "ttir.constant"() <{value = dense<[100, 200]> : tensor<2xsi32>}> : () -> tensor<2xsi32>
    %3 = "ttir.where"(%0, %1, %2) : (tensor<2xsi32>, tensor<2x1xsi32>, tensor<2xsi32>) -> tensor<2x2xsi32>
    return %3 : tensor<2x2xsi32>
  }
}

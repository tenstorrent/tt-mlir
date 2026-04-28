// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @add_float() -> tensor<3xf32> {
    // CHECK-LABEL: @add_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}-5.500000e-01, 2.001500e+02, 2.000000e+02]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.constant"() {value = dense<[0.0, 200.15, 201.15]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[-0.55, 0.0, -1.15]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.add"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @add_bf() -> tensor<3xbf16> {
    // CHECK-LABEL: @add_bf
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}-5.50{{[0-9]*}}e-01, 2.00{{[0-9]*}}e+02, 2.00{{[0-9]*}}e+02]> : tensor<3xbf16>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.constant"() {value = dense<[0.0, 200.15, 201.15]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %1 = "ttir.constant"() {value = dense<[-0.55, 0.0, -1.15]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %2 = "ttir.add"(%0, %1) : (tensor<3xbf16>, tensor<3xbf16>) -> tensor<3xbf16>
    return %2 : tensor<3xbf16>
  }

  func.func @add_sint() -> tensor<3xsi32> {
    // CHECK-LABEL: @add_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}-1, 200, 200]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.constant"() {value = dense<[0, 200, 201]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[-1, 0, -1]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.add"(%0, %1) : (tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %2 : tensor<3xsi32>
  }

  func.func @add_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @add_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}255, 5, 250]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.constant"() {value = dense<[0, 2, 201]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[255, 3, 49]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.add"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @add_full() -> tensor<3x3x3xsi32> {
    // CHECK-LABEL: @add_full
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 201
    // CHECK-NOT: "ttir.ones"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.full"() {fill_value = 200 : i32, shape = array<i32: 1, 3, 1>} : () -> tensor<1x3x1xsi32>
    %1 = "ttir.ones"() {shape = array<i32: 3, 1, 3>} : () -> tensor<3x1x3xsi32>
    %2 = "ttir.add"(%0, %1) : (tensor<1x3x1xsi32>, tensor<3x1x3xsi32>) -> tensor<3x3x3xsi32>
    return %2 : tensor<3x3x3xsi32>
  }

  func.func @add_full_and_constant() -> tensor<3xsi32> {
    // CHECK-LABEL: @add_full_and_constant
    // CHECK-NOT: "ttir.full"
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[201, 202, 203]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.full"() {fill_value = 200 : i32, shape = array<i32: 3>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[1, 2, 3]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.add"(%0, %1) : (tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %2 : tensor<3xsi32>
  }

  func.func @add_unmatched_type_no_fold() -> tensor<3xf32> {
    // CHECK-LABEL: @add_unmatched_type_no_fold
    // CHECK: "ttir.constant"
    // CHECK: "ttir.constant"
    // CHECK: "ttir.add"
    %0 = "ttir.constant"() {value = dense<[0, 200, 201]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[-1.0, 0.0, -1.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.add"(%0, %1) : (tensor<3xsi32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @add_broadcastible_non_splat_no_fold() -> tensor<3x3xf32> {
    // CHECK-LABEL: @add_broadcastible_non_splat_no_fold
    // CHECK: "ttir.constant"
    // CHECK: "ttir.constant"
    // CHECK: "ttir.add"
    %0 = "ttir.constant"() {value = dense<[[0.0], [200.15], [201.15]]> : tensor<3x1xf32>} : () -> tensor<3x1xf32>
    %1 = "ttir.constant"() {value = dense<[[-0.55, 0.0, -1.15]]> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
    %2 = "ttir.add"(%0, %1) : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
    return %2 : tensor<3x3xf32>
  }

  func.func @atan2() -> tensor<3xf32> {
    // CHECK-LABEL: @atan2
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}-0.78539{{[0-9]*(e\+00)?}}, 0.00000{{[0-9]*(e\+00)?}}, 0.78539{{[0-9]*(e\+00)?}}]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.atan2"
    %0 = "ttir.constant"() {value = dense<[-1.0, 0.0, 1.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[1.0, 1.0, 1.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.atan2"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @div_float() -> tensor<3xf32> {
    // CHECK-LABEL: @div_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.500000e+00, -2.250000e+00, -2.500000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.div"
    %0 = "ttir.constant"() {value = dense<[3.0, -4.5, 10.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[2.0, 2.0, -4.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.div"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @div_sint() -> tensor<3xsi32> {
    // CHECK-LABEL: @div_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}10, -5, 2]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.div"
    %0 = "ttir.constant"() {value = dense<[10, -20, -7]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[1, 4, -3]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.div"(%0, %1) : (tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %2 : tensor<3xsi32>
  }

  func.func @div_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @div_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}10, 5, 50]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.div"
    %0 = "ttir.constant"() {value = dense<[10, 20, 250]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[1, 4, 5]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.div"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @bitwise_and_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @bitwise_and_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1, 0, 0]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.bitwise_and"
    %0 = "ttir.constant"() {value = dense<[255, 170, 240]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[1, 85, 15]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.bitwise_and"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @bitwise_or_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @bitwise_or_uint
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 255
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.bitwise_or"
    %0 = "ttir.constant"() {value = dense<[255, 170, 240]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[1, 85, 15]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.bitwise_or"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @bitwise_xor_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @bitwise_xor_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}254, 255, 255]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.bitwise_xor"
    %0 = "ttir.constant"() {value = dense<[255, 170, 240]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[1, 85, 15]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.bitwise_xor"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @equal_float() -> tensor<3xf32> {
    // CHECK-LABEL: @equal_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.eq"
    %0 = "ttir.constant"() {value = dense<[1.0, 16.0, -2.5]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[1.0, 4.0, -2.5]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.eq"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @equal_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @equal_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1, 0, 1]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.eq"
    %0 = "ttir.constant"() {value = dense<[255, 170, 15]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[255, 85, 15]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.eq"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @not_equal_float() -> tensor<3xf32> {
    // CHECK-LABEL: @not_equal_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.ne"
    %0 = "ttir.constant"() {value = dense<[1.0, 16.0, -2.5]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[1.0, 4.0, 0.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.ne"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @not_equal_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @not_equal_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}0, 1, 0]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.ne"
    %0 = "ttir.constant"() {value = dense<[255, 170, 15]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[255, 85, 15]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.ne"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @greater_equal_float() -> tensor<3xf32> {
    // CHECK-LABEL: @greater_equal_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+00, 1.000000e+00, 0.000000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.ge"
    %0 = "ttir.constant"() {value = dense<[1.0, 16.0, -2.5]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[1.0, 4.0, 0.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.ge"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @greater_equal_sint() -> tensor<3xsi32> {
    // CHECK-LABEL: @greater_equal_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}0, 1, 0]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.ge"
    %0 = "ttir.constant"() {value = dense<[-1, -5, 7]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[1, -5, 8]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.ge"(%0, %1) : (tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %2 : tensor<3xsi32>
  }

  func.func @greater_equal_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @greater_equal_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1, 1, 0]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.ge"
    %0 = "ttir.constant"() {value = dense<[255, 170, 15]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[255, 85, 16]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.ge"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @greater_than_float() -> tensor<3xf32> {
    // CHECK-LABEL: @greater_than_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}0.000000e+00, 1.000000e+00, 0.000000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.gt"
    %0 = "ttir.constant"() {value = dense<[1.0, 16.0, -2.5]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[1.0, 4.0, 0.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.gt"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @greater_than_sint() -> tensor<3xsi32> {
    // CHECK-LABEL: @greater_than_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}0, 1, 0]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.gt"
    %0 = "ttir.constant"() {value = dense<[-1, -5, 7]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[1, -6, 8]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.gt"(%0, %1) : (tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %2 : tensor<3xsi32>
  }

  func.func @greater_than_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @greater_than_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}0, 1, 0]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.gt"
    %0 = "ttir.constant"() {value = dense<[255, 170, 15]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[255, 85, 16]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.gt"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @logical_and_float() -> tensor<4xf32> {
    // CHECK-LABEL: @logical_and_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<4xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.logical_and"
    %0 = "ttir.constant"() {value = dense<[1.0, -2.5, 0.0, 0.0]> : tensor<4xf32>} : () -> tensor<4xf32>
    %1 = "ttir.constant"() {value = dense<[4.0, 0.0, 3.0, 0.0]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "ttir.logical_and"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %2 : tensor<4xf32>
  }

  func.func @logical_and_uint() -> tensor<4xui8> {
    // CHECK-LABEL: @logical_and_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1, 0, 0, 0]> : tensor<4xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.logical_and"
    %0 = "ttir.constant"() {value = dense<[2, 2, 0, 0]> : tensor<4xui8>} : () -> tensor<4xui8>
    %1 = "ttir.constant"() {value = dense<[4, 0, 5, 0]> : tensor<4xui8>} : () -> tensor<4xui8>
    %2 = "ttir.logical_and"(%0, %1) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
    return %2 : tensor<4xui8>
  }

  func.func @logical_or_float() -> tensor<4xf32> {
    // CHECK-LABEL: @logical_or_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00]> : tensor<4xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.logical_or"
    %0 = "ttir.constant"() {value = dense<[1.0, -2.5, 0.0, 0.0]> : tensor<4xf32>} : () -> tensor<4xf32>
    %1 = "ttir.constant"() {value = dense<[4.0, 0.0, 3.0, 0.0]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "ttir.logical_or"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %2 : tensor<4xf32>
  }

  func.func @logical_or_uint() -> tensor<4xui8> {
    // CHECK-LABEL: @logical_or_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1, 1, 1, 0]> : tensor<4xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.logical_or"
    %0 = "ttir.constant"() {value = dense<[2, 2, 0, 0]> : tensor<4xui8>} : () -> tensor<4xui8>
    %1 = "ttir.constant"() {value = dense<[4, 0, 5, 0]> : tensor<4xui8>} : () -> tensor<4xui8>
    %2 = "ttir.logical_or"(%0, %1) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
    return %2 : tensor<4xui8>
  }

  func.func @logical_left_shift_sint() -> tensor<4xsi32> {
    // CHECK-LABEL: @logical_left_shift_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}2, -4, 112, 0]> : tensor<4xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.logical_left_shift"
    %0 = "ttir.constant"() {value = dense<[1, -2, 7, -1]> : tensor<4xsi32>} : () -> tensor<4xsi32>
    %1 = "ttir.constant"() {value = dense<[1, 1, 4, 32]> : tensor<4xsi32>} : () -> tensor<4xsi32>
    %2 = "ttir.logical_left_shift"(%0, %1) : (tensor<4xsi32>, tensor<4xsi32>) -> tensor<4xsi32>
    return %2 : tensor<4xsi32>
  }

  func.func @logical_left_shift_uint() -> tensor<4xui8> {
    // CHECK-LABEL: @logical_left_shift_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}2, 12, 240, 0]> : tensor<4xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.logical_left_shift"
    %0 = "ttir.constant"() {value = dense<[1, 3, 255, 1]> : tensor<4xui8>} : () -> tensor<4xui8>
    %1 = "ttir.constant"() {value = dense<[1, 2, 4, 8]> : tensor<4xui8>} : () -> tensor<4xui8>
    %2 = "ttir.logical_left_shift"(%0, %1) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
    return %2 : tensor<4xui8>
  }

  func.func @logical_right_shift_sint() -> tensor<3xsi32> {
    // CHECK-LABEL: @logical_right_shift_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}2147483647, 3, 0]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.logical_right_shift"
    %0 = "ttir.constant"() {value = dense<[-1, 7, 3]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[1, 1, 32]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.logical_right_shift"(%0, %1) : (tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %2 : tensor<3xsi32>
  }

  func.func @logical_right_shift_uint() -> tensor<4xui8> {
    // CHECK-LABEL: @logical_right_shift_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}64, 3, 15, 0]> : tensor<4xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.logical_right_shift"
    %0 = "ttir.constant"() {value = dense<[128, 7, 255, 3]> : tensor<4xui8>} : () -> tensor<4xui8>
    %1 = "ttir.constant"() {value = dense<[1, 1, 4, 8]> : tensor<4xui8>} : () -> tensor<4xui8>
    %2 = "ttir.logical_right_shift"(%0, %1) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
    return %2 : tensor<4xui8>
  }

  func.func @maximum_float() -> tensor<3xf32> {
    // CHECK-LABEL: @maximum_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+00, 4.000000e+00, 0.000000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.maximum"
    %0 = "ttir.constant"() {value = dense<[1.0, -2.5, 0.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[0.5, 4.0, -3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.maximum"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @maximum_sint() -> tensor<3xsi32> {
    // CHECK-LABEL: @maximum_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}-1, -5, 8]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.maximum"
    %0 = "ttir.constant"() {value = dense<[-1, -5, 7]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[-2, -6, 8]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.maximum"(%0, %1) : (tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %2 : tensor<3xsi32>
  }

  func.func @maximum_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @maximum_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}255, 170, 16]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.maximum"
    %0 = "ttir.constant"() {value = dense<[255, 85, 15]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[1, 170, 16]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.maximum"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @minimum_float() -> tensor<3xf32> {
    // CHECK-LABEL: @minimum_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}5.000000e-01, -2.500000e+00, -3.000000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.minimum"
    %0 = "ttir.constant"() {value = dense<[1.0, -2.5, 0.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[0.5, 4.0, -3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.minimum"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @minimum_sint() -> tensor<3xsi32> {
    // CHECK-LABEL: @minimum_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}-2, -6, 7]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.minimum"
    %0 = "ttir.constant"() {value = dense<[-1, -5, 7]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[-2, -6, 8]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.minimum"(%0, %1) : (tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %2 : tensor<3xsi32>
  }

  func.func @minimum_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @minimum_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1, 85, 15]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.minimum"
    %0 = "ttir.constant"() {value = dense<[255, 85, 15]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[1, 170, 16]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.minimum"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @multiply_float() -> tensor<3xf32> {
    // CHECK-LABEL: @multiply_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.000000e+00, -1.000000e+01, -4.500000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.multiply"
    %0 = "ttir.constant"() {value = dense<[0.5, -2.5, 3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[2.0, 4.0, -1.5]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.multiply"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @multiply_sint() -> tensor<3xsi32> {
    // CHECK-LABEL: @multiply_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}-8, -21, 45]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.multiply"
    %0 = "ttir.constant"() {value = dense<[2, 3, -5]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[-4, -7, -9]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.multiply"(%0, %1) : (tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %2 : tensor<3xsi32>
  }

  func.func @pow_float() -> tensor<3xf32> {
    // CHECK-LABEL: @pow_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}4.000000e+00, 9.000000e+00, 2.000000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.pow"
    %0 = "ttir.constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[2.0, 2.0, 0.5]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.pow"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @remainder_float() -> tensor<3xf32> {
    // CHECK-LABEL: @remainder_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}-2.250000e+00, 7.500000e-01, -1.40000{{[0-9]*(e\+00)?}}]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.remainder"
    %0 = "ttir.constant"() {value = dense<[-5.25, -5.25, 5.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[-3.0, 3.0, -3.2]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.remainder"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @remainder_sint() -> tensor<6xsi32> {
    // CHECK-LABEL: @remainder_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}-2,  1, -1,  2,  0,  0]> : tensor<6xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.remainder"
    %0 = "ttir.constant"() <{value = dense<[-5, -5, 5, 5, 5, -5]> : tensor<6xsi32>}> : () -> tensor<6xsi32>
    %1 = "ttir.constant"() <{value = dense<[-3, 3, -3, 3, 5, 1]> : tensor<6xsi32>}> : () -> tensor<6xsi32>
    %2 = "ttir.remainder"(%0, %1) : (tensor<6xsi32>, tensor<6xsi32>) -> tensor<6xsi32>
    return %2 : tensor<6xsi32>
  }

  func.func @remainder_uint() -> tensor<3xui8> {
    // CHECK-LABEL: @remainder_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}2, 3, 3]> : tensor<3xui8>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.remainder"
    %0 = "ttir.constant"() {value = dense<[17, 18, 19]> : tensor<3xui8>} : () -> tensor<3xui8>
    %1 = "ttir.constant"() {value = dense<[3, 5, 4]> : tensor<3xui8>} : () -> tensor<3xui8>
    %2 = "ttir.remainder"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    return %2 : tensor<3xui8>
  }

  func.func @subtract_float() -> tensor<3xf32> {
    // CHECK-LABEL: @subtract_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}2.000000e+00, -2.000000e+00, 2.000000e+00]> : tensor<3xf32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.subtract"
    %0 = "ttir.constant"() {value = dense<[3.5, 0.0, -1.2]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "ttir.constant"() {value = dense<[1.5, 2.0, -3.2]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "ttir.subtract"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }

  func.func @subtract_sint() -> tensor<3xsi32> {
    // CHECK-LABEL: @subtract_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}9, -18, 27]> : tensor<3xsi32>
    // CHECK-NOT: "ttir.constant"
    // CHECK-NOT: "ttir.subtract"
    %0 = "ttir.constant"() {value = dense<[10, -20, 30]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %1 = "ttir.constant"() {value = dense<[1, -2, 3]> : tensor<3xsi32>} : () -> tensor<3xsi32>
    %2 = "ttir.subtract"(%0, %1) : (tensor<3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
    return %2 : tensor<3xsi32>
  }
}

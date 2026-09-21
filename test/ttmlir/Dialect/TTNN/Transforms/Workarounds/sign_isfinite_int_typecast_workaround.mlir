// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that integer inputs to ttnn.sign and ttnn.isfinite are typecast to
// bf16 around the op. Since tt-metal #56980 these float-only SFPU kernels are
// rejected outright for integer dtypes (`unary_op_supports_integer_dtype` in
// unary_device_operation.cpp), so without the typecast the op hard-faults.
//
// bf16 is exact for both: `sign` only needs which side of zero a value falls
// on, which rounding cannot change, and every int32 maps to a finite bf16 so
// `isfinite` is unconditionally true - which is the correct answer.

module {
  func.func public @test_sign_i32_to_bf16(%arg0: tensor<128x128xsi32>) -> tensor<128x128xsi32> {
    // CHECK-LABEL: func.func public @test_sign_i32_to_bf16
    // CHECK: "ttnn.to_tensor_spec"
    // CHECK-SAME: -> tensor<{{.*}}xbf16,
    // CHECK: "ttnn.sign"
    // CHECK-SAME: tensor<128x128xbf16
    // CHECK-SAME: -> tensor<128x128xbf16
    // CHECK: "ttnn.to_tensor_spec"
    %0 = "ttir.sign"(%arg0) : (tensor<128x128xsi32>) -> tensor<128x128xsi32>
    return %0 : tensor<128x128xsi32>
  }

  func.func public @test_isfinite_i32_to_bf16(%arg0: tensor<128x128xsi32>) -> tensor<128x128xsi32> {
    // CHECK-LABEL: func.func public @test_isfinite_i32_to_bf16
    // CHECK: "ttnn.to_tensor_spec"
    // CHECK-SAME: -> tensor<{{.*}}xbf16,
    // CHECK: "ttnn.isfinite"
    // CHECK-SAME: tensor<128x128xbf16
    // CHECK-SAME: -> tensor<128x128xbf16
    // CHECK: "ttnn.to_tensor_spec"
    %0 = "ttir.isfinite"(%arg0) : (tensor<128x128xsi32>) -> tensor<128x128xsi32>
    return %0 : tensor<128x128xsi32>
  }

  func.func public @test_sign_f32_no_workaround(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // Float inputs should NOT trigger a dtype workaround.
    // CHECK-LABEL: func.func public @test_sign_f32_no_workaround
    // CHECK-NOT: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.sign"
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: -> tensor<64x128xf32
    %0 = "ttir.sign"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  func.func public @test_isfinite_bf16_no_workaround(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // bf16 inputs should also pass through unchanged.
    // CHECK-LABEL: func.func public @test_isfinite_bf16_no_workaround
    // CHECK: "ttnn.isfinite"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: -> tensor<64x128xbf16
    %0 = "ttir.isfinite"(%arg0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }
}

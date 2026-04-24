// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that integer inputs to ttnn.erf are typecast to bf16 around the op.
// The tt-metal LUT-based erf SFPU kernel reads the input as a float; without a
// typecast workaround integer bit patterns become NaN/Inf and the result no
// longer matches torch.erf.

module {
  func.func public @test_erf_i32_to_bf16(%arg0: tensor<128x128xsi32>) -> tensor<128x128xsi32> {
    // CHECK-LABEL: func.func public @test_erf_i32_to_bf16
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.erf"
    // CHECK-SAME: tensor<128x128xbf16
    // CHECK-SAME: -> tensor<128x128xbf16
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    %0 = "ttir.erf"(%arg0) : (tensor<128x128xsi32>) -> tensor<128x128xsi32>
    return %0 : tensor<128x128xsi32>
  }

  func.func public @test_erf_f32_no_workaround(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // Float inputs should NOT trigger a dtype workaround.
    // CHECK-LABEL: func.func public @test_erf_f32_no_workaround
    // CHECK-NOT: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.erf"
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: -> tensor<64x128xf32
    %0 = "ttir.erf"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  func.func public @test_erf_bf16_no_workaround(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // bf16 inputs should also pass through unchanged.
    // CHECK-LABEL: func.func public @test_erf_bf16_no_workaround
    // CHECK: "ttnn.erf"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: -> tensor<64x128xbf16
    %0 = "ttir.erf"(%arg0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }
}

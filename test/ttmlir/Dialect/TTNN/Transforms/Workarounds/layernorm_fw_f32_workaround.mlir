// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround="ttnn-optimization-level=1" --canonicalize -o %t1 %s
// RUN: FileCheck %s --check-prefix=OPT1 --input-file=%t1

// Test that at opt-level 0 f32 operands of ttml layernorm_fw are automatically
// converted to bf16 (and the bf16 results cast back to f32) by the workaround
// pass, since the backing metal kernel only accepts bf16.

module {
  // f32 input/weight/bias, no backward statistics.
  func.func public @layernorm_fw_f32(%input: tensor<1x1x128x256xf32>, %weight: tensor<1x1x1x256xf32>, %bias: tensor<1x1x1x256xf32>) -> tensor<1x1x128x256xf32> {
    // CHECK-LABEL: func.func public @layernorm_fw_f32
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg1)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg2)
    // CHECK: %[[OUT_BF16:.*]] = "ttnn.layernorm_fw"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x128x256xbf16
    // CHECK: "ttnn.to_tensor_spec"(%[[OUT_BF16]])
    // CHECK-SAME: -> tensor<1x1x128x256xf32

    // At opt-level 1 the workaround is skipped: no bf16 casts, op stays f32.
    // OPT1-LABEL: func.func public @layernorm_fw_f32
    // OPT1-NOT: ttnn.to_tensor_spec
    // OPT1: "ttnn.layernorm_fw"
    // OPT1-SAME: -> tensor<1x1x128x256xf32
    %0 = "ttnn.layernorm_fw"(%input, %weight, %bias) <{epsilon = 1.000000e-05 : f32, return_mean_rstd = false}> : (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x128x256xf32>
    return %0 : tensor<1x1x128x256xf32>
  }

  // f32 operands returning mean/rstd: all three results are cast back to f32.
  func.func public @layernorm_fw_f32_mean_rstd(%input: tensor<1x1x128x256xf32>, %weight: tensor<1x1x1x256xf32>, %bias: tensor<1x1x1x256xf32>) -> (tensor<1x1x128x256xf32>, tensor<1x1x128x1xf32>, tensor<1x1x128x1xf32>) {
    // CHECK-LABEL: func.func public @layernorm_fw_f32_mean_rstd
    // CHECK: %[[OUT_BF16:.*]], %[[MEAN_BF16:.*]], %[[RSTD_BF16:.*]] = "ttnn.layernorm_fw"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16{{.*}}, tensor<1x1x128x1xbf16{{.*}}, tensor<1x1x128x1xbf16
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[OUT_BF16]])
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[MEAN_BF16]])
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[RSTD_BF16]])
    %0, %1, %2 = "ttnn.layernorm_fw"(%input, %weight, %bias) <{epsilon = 1.000000e-05 : f32, return_mean_rstd = true}> : (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> (tensor<1x1x128x256xf32>, tensor<1x1x128x1xf32>, tensor<1x1x128x1xf32>)
    return %0, %1, %2 : tensor<1x1x128x256xf32>, tensor<1x1x128x1xf32>, tensor<1x1x128x1xf32>
  }

  // bf16 operands should not trigger the workaround.
  func.func public @layernorm_fw_bf16_no_workaround(%input: tensor<1x1x128x256xbf16>, %weight: tensor<1x1x1x256xbf16>, %bias: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // CHECK-LABEL: func.func public @layernorm_fw_bf16_no_workaround
    // CHECK-NOT: ttnn.to_tensor_spec
    // CHECK: "ttnn.layernorm_fw"
    %0 = "ttnn.layernorm_fw"(%input, %weight, %bias) <{epsilon = 1.000000e-05 : f32, return_mean_rstd = false}> : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }
}

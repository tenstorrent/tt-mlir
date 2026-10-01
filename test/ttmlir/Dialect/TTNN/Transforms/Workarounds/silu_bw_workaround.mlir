// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Convert every silu_bw operand and result to the tiled, DRAM-interleaved bf16
// layout required by the backing metal kernel.

module {
  func.func public @silu_bw_f32(%input: tensor<1x1x128x256xf32>, %grad_output: tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32> {
    // CHECK-LABEL: func.func public @silu_bw_f32
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg1)
    // CHECK: %[[GRAD_INPUT_BF16:.*]] = "ttnn.silu_bw"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x128x256xbf16
    // CHECK: "ttnn.to_tensor_spec"(%[[GRAD_INPUT_BF16]])
    %0 = "ttnn.silu_bw"(%input, %grad_output) : (tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
    return %0 : tensor<1x1x128x256xf32>
  }

  func.func public @silu_bw_bf16_no_workaround(%input: tensor<1x1x128x256xbf16>, %grad_output: tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // CHECK-LABEL: func.func public @silu_bw_bf16_no_workaround
    // CHECK-NOT: ttnn.to_tensor_spec
    // CHECK: "ttnn.silu_bw"
    // CHECK-NOT: ttnn.to_tensor_spec
    // CHECK: return
    %0 = "ttnn.silu_bw"(%input, %grad_output) : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }
}

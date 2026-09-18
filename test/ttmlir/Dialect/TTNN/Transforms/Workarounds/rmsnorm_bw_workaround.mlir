// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Convert every rmsnorm_bw operand and result to the tiled, DRAM-interleaved
// bf16 layout required by the backing metal kernel.

module {
  func.func public @rmsnorm_bw_f32(%input: tensor<1x1x128x256xf32>, %gamma: tensor<1x1x1x256xf32>, %rms: tensor<1x1x128x1xf32>, %grad_output: tensor<1x1x128x256xf32>) -> (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>) {
    // CHECK-LABEL: func.func public @rmsnorm_bw_f32
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg1)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg2)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg3)
    // CHECK: %[[DX_BF16:.*]], %[[DG_BF16:.*]] = "ttnn.rmsnorm_bw"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16{{.*}}, tensor<1x1x1x256xbf16
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[DX_BF16]])
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[DG_BF16]])
    %0, %1 = "ttnn.rmsnorm_bw"(%input, %gamma, %rms, %grad_output) : (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>, tensor<1x1x128x1xf32>, tensor<1x1x128x256xf32>) -> (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>)
    return %0, %1 : tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>
  }

  func.func public @rmsnorm_bw_bf16_no_workaround(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>, %rms: tensor<1x1x128x1xbf16>, %grad_output: tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    // CHECK-LABEL: func.func public @rmsnorm_bw_bf16_no_workaround
    // CHECK-NOT: ttnn.to_tensor_spec
    // CHECK: "ttnn.rmsnorm_bw"
    // CHECK-NOT: ttnn.to_tensor_spec
    // CHECK: return
    %0, %1 = "ttnn.rmsnorm_bw"(%input, %gamma, %rms, %grad_output) : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
    return %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }
}

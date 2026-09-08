// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround="ttnn-optimization-level=1" --canonicalize -o %t1 %s
// RUN: FileCheck %s --check-prefix=OPT1 --input-file=%t1

// At optimization level 0, convert every rmsnorm_fw operand and result to the
// tiled, DRAM-interleaved bf16 layout required by the backing metal kernel.

module {
  func.func public @rmsnorm_fw_f32(%input: tensor<1x1x128x256xf32>, %gamma: tensor<1x1x1x256xf32>) -> tensor<1x1x128x256xf32> {
    // CHECK-LABEL: func.func public @rmsnorm_fw_f32
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-DAG: "ttnn.to_tensor_spec"(%arg1)
    // CHECK: %[[OUT_BF16:.*]] = "ttnn.rmsnorm_fw"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x128x256xbf16
    // CHECK: "ttnn.to_tensor_spec"(%[[OUT_BF16]])
    // CHECK-SAME: -> tensor<1x1x128x256xf32

    // At optimization level 1 the workaround is skipped.
    // OPT1-LABEL: func.func public @rmsnorm_fw_f32
    // OPT1-NOT: ttnn.to_tensor_spec
    // OPT1: "ttnn.rmsnorm_fw"
    // OPT1-SAME: -> tensor<1x1x128x256xf32
    %0 = "ttnn.rmsnorm_fw"(%input, %gamma) <{epsilon = 1.000000e-06 : f32, return_intermediates = false}> : (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x128x256xf32>
    return %0 : tensor<1x1x128x256xf32>
  }

  func.func public @rmsnorm_fw_f32_rms(%input: tensor<1x1x128x256xf32>, %gamma: tensor<1x1x1x256xf32>) -> (tensor<1x1x128x256xf32>, tensor<1x1x128x1xf32>) {
    // CHECK-LABEL: func.func public @rmsnorm_fw_f32_rms
    // CHECK: %[[OUT_BF16:.*]], %[[RMS_BF16:.*]] = "ttnn.rmsnorm_fw"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16{{.*}}, tensor<1x1x128x1xbf16
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[OUT_BF16]])
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[RMS_BF16]])
    %0, %1 = "ttnn.rmsnorm_fw"(%input, %gamma) <{epsilon = 1.000000e-06 : f32, return_intermediates = true}> : (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>) -> (tensor<1x1x128x256xf32>, tensor<1x1x128x1xf32>)
    return %0, %1 : tensor<1x1x128x256xf32>, tensor<1x1x128x1xf32>
  }

  func.func public @rmsnorm_fw_bf16_no_workaround(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // CHECK-LABEL: func.func public @rmsnorm_fw_bf16_no_workaround
    // CHECK-NOT: ttnn.to_tensor_spec
    // CHECK: "ttnn.rmsnorm_fw"
    // CHECK-NOT: ttnn.to_tensor_spec
    // CHECK: return
    %0 = "ttnn.rmsnorm_fw"(%input, %gamma) <{epsilon = 1.000000e-06 : f32, return_intermediates = false}> : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }
}

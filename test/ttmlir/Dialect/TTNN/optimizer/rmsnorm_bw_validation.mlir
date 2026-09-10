// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // The TTML kernel requires tiled, DRAM-interleaved operands, and the trailing
  // ttnn::sum that reduces grad_gamma inherits gamma's memory config. The
  // backend derives both output layouts from the inputs.
  func.func @rmsnorm_bw(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>,
      %rms: tensor<1x1x128x1xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    // CHECK-DAG: #[[INPUT_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[GAMMA_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[RMS_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

    // CHECK-LABEL: func.func @rmsnorm_bw(
    // CHECK-SAME: %[[INPUT:[0-9a-z_]+]]: tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>
    // CHECK-SAME: %[[GAMMA:[0-9a-z_]+]]: tensor<1x1x1x256xbf16, #[[GAMMA_LAYOUT]]>
    // CHECK-SAME: %[[RMS:[0-9a-z_]+]]: tensor<1x1x128x1xbf16, #[[RMS_LAYOUT]]>
    // CHECK-SAME: %[[GRAD_OUTPUT:[0-9a-z_]+]]: tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>)
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>, tensor<1x1x1x256xbf16, #[[GAMMA_LAYOUT]]>)
    // CHECK: %[[GRAD_INPUT:[0-9a-z_]+]], %[[GRAD_GAMMA:[0-9a-z_]+]] = "ttnn.rmsnorm_bw"(%[[INPUT]], %[[GAMMA]], %[[RMS]], %[[GRAD_OUTPUT]])
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>, tensor<1x1x1x256xbf16, #[[GAMMA_LAYOUT]]>)
    // CHECK: return %[[GRAD_INPUT]], %[[GRAD_GAMMA]]
    %grad_input, %grad_gamma = "ttcore.composite"(%input, %gamma, %rms, %grad_output) <{
        composite_name = "rmsnorm_bw",
        decomposition = @rmsnorm_bw_decomp}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>,
           tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
    return %grad_input, %grad_gamma
        : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }

  func.func @rmsnorm_bw_f32(
      %input: tensor<1x1x128x256xf32>,
      %gamma: tensor<1x1x1x256xf32>,
      %rms: tensor<1x1x128x1xf32>,
      %grad_output: tensor<1x1x128x256xf32>)
      -> (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>) {
    // CHECK-LABEL: func.func @rmsnorm_bw_f32(
    // CHECK: %[[INPUT_BF16:[0-9a-z_]+]] = "ttnn.typecast"(%{{.*}})
    // CHECK-SAME: -> tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>
    // CHECK: %[[GAMMA_BF16:[0-9a-z_]+]] = "ttnn.typecast"(%{{.*}})
    // CHECK-SAME: -> tensor<1x1x1x256xbf16, #[[GAMMA_LAYOUT]]>
    // CHECK: %[[RMS_BF16:[0-9a-z_]+]] = "ttnn.typecast"(%{{.*}})
    // CHECK-SAME: -> tensor<1x1x128x1xbf16, #[[RMS_LAYOUT]]>
    // CHECK: %[[GRAD_OUTPUT_BF16:[0-9a-z_]+]] = "ttnn.typecast"(%{{.*}})
    // CHECK-SAME: -> tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>
    // CHECK: %[[GRAD_INPUT_BF16:[0-9a-z_]+]], %[[GRAD_GAMMA_BF16:[0-9a-z_]+]] = "ttnn.rmsnorm_bw"(%[[INPUT_BF16]], %[[GAMMA_BF16]], %[[RMS_BF16]], %[[GRAD_OUTPUT_BF16]])
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>, tensor<1x1x1x256xbf16, #[[GAMMA_LAYOUT]]>)
    // CHECK-DAG: %[[GRAD_INPUT_F32:[0-9a-z_]+]] = "ttnn.typecast"(%[[GRAD_INPUT_BF16]])
    // CHECK-DAG: %[[GRAD_GAMMA_F32:[0-9a-z_]+]] = "ttnn.typecast"(%[[GRAD_GAMMA_BF16]])
    // CHECK: return %[[GRAD_INPUT_F32]], %[[GRAD_GAMMA_F32]]
    %grad_input, %grad_gamma = "ttcore.composite"(
        %input, %gamma, %rms, %grad_output) <{
        composite_name = "rmsnorm_bw",
        decomposition = @rmsnorm_bw_f32_decomp}>
        : (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>,
           tensor<1x1x128x1xf32>, tensor<1x1x128x256xf32>)
          -> (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>)
    return %grad_input, %grad_gamma
        : tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>
  }

  func.func private @rmsnorm_bw_decomp(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>,
      %rms: tensor<1x1x128x1xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    return %input, %gamma
        : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }

  func.func private @rmsnorm_bw_f32_decomp(
      %input: tensor<1x1x128x256xf32>,
      %gamma: tensor<1x1x1x256xf32>,
      %rms: tensor<1x1x128x1xf32>,
      %grad_output: tensor<1x1x128x256xf32>)
      -> (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>) {
    return %input, %gamma
        : tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>
  }
}

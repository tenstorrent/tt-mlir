// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // The TTML kernel requires tiled, DRAM-interleaved operands and derives the
  // gradient layout from the input.
  func.func @silu_bw(
      %input: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> tensor<1x1x128x256xbf16> {
    // CHECK-DAG: #[[LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

    // CHECK-LABEL: func.func @silu_bw(
    // CHECK-SAME: %[[INPUT:[0-9a-z_]+]]: tensor<1x1x128x256xbf16, #[[LAYOUT]]>
    // CHECK-SAME: %[[GRAD_OUTPUT:[0-9a-z_]+]]: tensor<1x1x128x256xbf16, #[[LAYOUT]]>)
    // CHECK-SAME: -> tensor<1x1x128x256xbf16, #[[LAYOUT]]>
    // CHECK: %[[GRAD_INPUT:[0-9a-z_]+]] = "ttnn.silu_bw"(%[[INPUT]], %[[GRAD_OUTPUT]])
    // CHECK-SAME: -> tensor<1x1x128x256xbf16, #[[LAYOUT]]>
    // CHECK: return %[[GRAD_INPUT]]
    %grad_input = "ttcore.composite"(%input, %grad_output) <{
        composite_name = "silu_bw",
        decomposition = @silu_bw_decomp}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
          -> tensor<1x1x128x256xbf16>
    return %grad_input : tensor<1x1x128x256xbf16>
  }

  func.func @silu_bw_f32(
      %input: tensor<1x1x128x256xf32>,
      %grad_output: tensor<1x1x128x256xf32>)
      -> tensor<1x1x128x256xf32> {
    // CHECK-LABEL: func.func @silu_bw_f32(
    // CHECK-DAG: %[[INPUT_BF16:[0-9a-z_]+]] = "ttnn.typecast"(%{{.*}})
    // CHECK-SAME: -> tensor<1x1x128x256xbf16, #[[LAYOUT]]>
    // CHECK-DAG: %[[GRAD_OUTPUT_BF16:[0-9a-z_]+]] = "ttnn.typecast"(%{{.*}})
    // CHECK-SAME: -> tensor<1x1x128x256xbf16, #[[LAYOUT]]>
    // CHECK: %[[GRAD_INPUT_BF16:[0-9a-z_]+]] = "ttnn.silu_bw"(%[[INPUT_BF16]], %[[GRAD_OUTPUT_BF16]])
    // CHECK-SAME: -> tensor<1x1x128x256xbf16, #[[LAYOUT]]>
    // CHECK: %[[GRAD_INPUT_F32:[0-9a-z_]+]] = "ttnn.typecast"(%[[GRAD_INPUT_BF16]])
    // CHECK: return %[[GRAD_INPUT_F32]]
    %grad_input = "ttcore.composite"(%input, %grad_output) <{
        composite_name = "silu_bw",
        decomposition = @silu_bw_f32_decomp}>
        : (tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>)
          -> tensor<1x1x128x256xf32>
    return %grad_input : tensor<1x1x128x256xf32>
  }

  func.func private @silu_bw_decomp(
      %input: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> tensor<1x1x128x256xbf16> {
    return %input : tensor<1x1x128x256xbf16>
  }

  func.func private @silu_bw_f32_decomp(
      %input: tensor<1x1x128x256xf32>,
      %grad_output: tensor<1x1x128x256xf32>)
      -> tensor<1x1x128x256xf32> {
    return %input : tensor<1x1x128x256xf32>
  }
}

// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // The TTML kernel requires tiled, interleaved operands and derives both
  // gradient layouts from the input.
  func.func @swiglu_elemwise_bw(
      %input: tensor<1x1x128x256xbf16>,
      %gate: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) {
    // CHECK-DAG: #[[LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

    // CHECK-LABEL: func.func @swiglu_elemwise_bw(
    // CHECK-SAME: %[[INPUT:[0-9a-z_]+]]: tensor<1x1x128x256xbf16, #[[LAYOUT]]>
    // CHECK-SAME: %[[GATE:[0-9a-z_]+]]: tensor<1x1x128x256xbf16, #[[LAYOUT]]>
    // CHECK-SAME: %[[GRAD_OUTPUT:[0-9a-z_]+]]: tensor<1x1x128x256xbf16, #[[LAYOUT]]>)
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16, #[[LAYOUT]]>, tensor<1x1x128x256xbf16, #[[LAYOUT]]>)
    // CHECK: %[[GRAD_INPUT:[0-9a-z_]+]], %[[GRAD_GATE:[0-9a-z_]+]] = "ttnn.swiglu_elemwise_bw"(%[[INPUT]], %[[GATE]], %[[GRAD_OUTPUT]])
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16, #[[LAYOUT]]>, tensor<1x1x128x256xbf16, #[[LAYOUT]]>)
    // CHECK: return %[[GRAD_INPUT]], %[[GRAD_GATE]]
    %grad_input, %grad_gate = "ttcore.composite"(%input, %gate, %grad_output) <{
        composite_name = "swiglu_elemwise_bw",
        decomposition = @swiglu_elemwise_bw_decomp}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
    return %grad_input, %grad_gate
        : tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }

  func.func @swiglu_elemwise_bw_f32(
      %input: tensor<1x1x128x256xf32>,
      %gate: tensor<1x1x128x256xf32>,
      %grad_output: tensor<1x1x128x256xf32>)
      -> (tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>) {
    // CHECK-LABEL: func.func @swiglu_elemwise_bw_f32(
    // CHECK: %[[INPUT_BF16:[0-9a-z_]+]] = "ttnn.typecast"(%{{.*}})
    // CHECK-SAME: -> tensor<1x1x128x256xbf16, #[[LAYOUT]]>
    // CHECK: %[[GRAD_INPUT_BF16:[0-9a-z_]+]], %[[GRAD_GATE_BF16:[0-9a-z_]+]] = "ttnn.swiglu_elemwise_bw"
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16, #[[LAYOUT]]>, tensor<1x1x128x256xbf16, #[[LAYOUT]]>)
    // CHECK-DAG: %[[GRAD_INPUT_F32:[0-9a-z_]+]] = "ttnn.typecast"(%[[GRAD_INPUT_BF16]])
    // CHECK-DAG: %[[GRAD_GATE_F32:[0-9a-z_]+]] = "ttnn.typecast"(%[[GRAD_GATE_BF16]])
    // CHECK: return %[[GRAD_INPUT_F32]], %[[GRAD_GATE_F32]]
    %grad_input, %grad_gate = "ttcore.composite"(%input, %gate, %grad_output) <{
        composite_name = "swiglu_elemwise_bw",
        decomposition = @swiglu_elemwise_bw_f32_decomp}>
        : (tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>)
          -> (tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>)
    return %grad_input, %grad_gate
        : tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>
  }

  func.func private @swiglu_elemwise_bw_decomp(
      %input: tensor<1x1x128x256xbf16>,
      %gate: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) {
    return %input, %gate
        : tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }

  func.func private @swiglu_elemwise_bw_f32_decomp(
      %input: tensor<1x1x128x256xf32>,
      %gate: tensor<1x1x128x256xf32>,
      %grad_output: tensor<1x1x128x256xf32>)
      -> (tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>) {
    return %input, %gate
        : tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>
  }
}

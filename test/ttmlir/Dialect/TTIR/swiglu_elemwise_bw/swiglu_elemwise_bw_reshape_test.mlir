// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s -o %t
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @swiglu_elemwise_bw_rank2
  func.func @swiglu_elemwise_bw_rank2(
      %input: tensor<128x256xbf16>,
      %gate: tensor<128x256xbf16>,
      %grad_output: tensor<128x256xbf16>)
      -> (tensor<128x256xbf16>, tensor<128x256xbf16>) {
    // CHECK-DAG: "ttir.reshape"(%arg0) {{.*}} -> tensor<1x1x128x256xbf16>
    // CHECK-DAG: "ttir.reshape"(%arg1) {{.*}} -> tensor<1x1x128x256xbf16>
    // CHECK-DAG: "ttir.reshape"(%arg2) {{.*}} -> tensor<1x1x128x256xbf16>
    // CHECK: "ttcore.composite"
    // CHECK-SAME: decomposition = @swiglu_elemwise_bw_rank2_decomp_rank4
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
    // CHECK-COUNT-2: "ttir.reshape"({{.*}}) {{.*}} -> tensor<128x256xbf16>
    %grad_input, %grad_gate = "ttcore.composite"(%input, %gate, %grad_output) <{
        composite_name = "swiglu_elemwise_bw",
        decomposition = @swiglu_elemwise_bw_rank2_decomp}>
        : (tensor<128x256xbf16>, tensor<128x256xbf16>, tensor<128x256xbf16>)
          -> (tensor<128x256xbf16>, tensor<128x256xbf16>)
    return %grad_input, %grad_gate : tensor<128x256xbf16>, tensor<128x256xbf16>
  }

  func.func private @swiglu_elemwise_bw_rank2_decomp(
      %input: tensor<128x256xbf16>,
      %gate: tensor<128x256xbf16>,
      %grad_output: tensor<128x256xbf16>)
      -> (tensor<128x256xbf16>, tensor<128x256xbf16>) {
    return %input, %gate : tensor<128x256xbf16>, tensor<128x256xbf16>
  }

  // CHECK-LABEL: func.func @swiglu_elemwise_bw_rank4
  func.func @swiglu_elemwise_bw_rank4(
      %input: tensor<2x4x128x256xbf16>,
      %gate: tensor<2x4x128x256xbf16>,
      %grad_output: tensor<2x4x128x256xbf16>)
      -> (tensor<2x4x128x256xbf16>, tensor<2x4x128x256xbf16>) {
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttcore.composite"
    // CHECK-SAME: decomposition = @swiglu_elemwise_bw_rank4_decomp
    // CHECK-NOT: "ttir.reshape"
    // CHECK: return
    %grad_input, %grad_gate = "ttcore.composite"(%input, %gate, %grad_output) <{
        composite_name = "swiglu_elemwise_bw",
        decomposition = @swiglu_elemwise_bw_rank4_decomp}>
        : (tensor<2x4x128x256xbf16>, tensor<2x4x128x256xbf16>, tensor<2x4x128x256xbf16>)
          -> (tensor<2x4x128x256xbf16>, tensor<2x4x128x256xbf16>)
    return %grad_input, %grad_gate
        : tensor<2x4x128x256xbf16>, tensor<2x4x128x256xbf16>
  }

  func.func private @swiglu_elemwise_bw_rank4_decomp(
      %input: tensor<2x4x128x256xbf16>,
      %gate: tensor<2x4x128x256xbf16>,
      %grad_output: tensor<2x4x128x256xbf16>)
      -> (tensor<2x4x128x256xbf16>, tensor<2x4x128x256xbf16>) {
    return %input, %gate
        : tensor<2x4x128x256xbf16>, tensor<2x4x128x256xbf16>
  }

  // CHECK-LABEL: func.func @swiglu_elemwise_bw_rank5
  func.func @swiglu_elemwise_bw_rank5(
      %input: tensor<2x3x4x128x256xbf16>,
      %gate: tensor<2x3x4x128x256xbf16>,
      %grad_output: tensor<2x3x4x128x256xbf16>)
      -> (tensor<2x3x4x128x256xbf16>, tensor<2x3x4x128x256xbf16>) {
    // CHECK: "ttir.reshape"(%arg0) {{.*}} -> tensor<1x24x128x256xbf16>
    // CHECK: "ttcore.composite"
    // CHECK-SAME: -> (tensor<1x24x128x256xbf16>, tensor<1x24x128x256xbf16>)
    // CHECK-COUNT-2: "ttir.reshape"({{.*}}) {{.*}} -> tensor<2x3x4x128x256xbf16>
    %grad_input, %grad_gate = "ttcore.composite"(%input, %gate, %grad_output) <{
        composite_name = "swiglu_elemwise_bw",
        decomposition = @swiglu_elemwise_bw_rank5_decomp}>
        : (tensor<2x3x4x128x256xbf16>, tensor<2x3x4x128x256xbf16>, tensor<2x3x4x128x256xbf16>)
          -> (tensor<2x3x4x128x256xbf16>, tensor<2x3x4x128x256xbf16>)
    return %grad_input, %grad_gate
        : tensor<2x3x4x128x256xbf16>, tensor<2x3x4x128x256xbf16>
  }

  func.func private @swiglu_elemwise_bw_rank5_decomp(
      %input: tensor<2x3x4x128x256xbf16>,
      %gate: tensor<2x3x4x128x256xbf16>,
      %grad_output: tensor<2x3x4x128x256xbf16>)
      -> (tensor<2x3x4x128x256xbf16>, tensor<2x3x4x128x256xbf16>) {
    return %input, %gate
        : tensor<2x3x4x128x256xbf16>, tensor<2x3x4x128x256xbf16>
  }
}

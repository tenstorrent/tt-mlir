// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s -o %t
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @silu_bw_rank2
  func.func @silu_bw_rank2(
      %input: tensor<128x256xbf16>,
      %grad_output: tensor<128x256xbf16>)
      -> tensor<128x256xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) {{.*}} -> tensor<1x1x128x256xbf16>
    // CHECK-DAG: "ttir.reshape"(%arg1) {{.*}} -> tensor<1x1x128x256xbf16>
    // CHECK: "ttcore.composite"
    // CHECK-SAME: decomposition = @silu_bw_rank2_decomp_rank4
    // CHECK-SAME: -> tensor<1x1x128x256xbf16>
    // CHECK: "ttir.reshape"({{.*}}) {{.*}} -> tensor<128x256xbf16>
    %grad_input = "ttcore.composite"(%input, %grad_output) <{
        composite_name = "silu_bw",
        decomposition = @silu_bw_rank2_decomp}>
        : (tensor<128x256xbf16>, tensor<128x256xbf16>)
          -> tensor<128x256xbf16>
    return %grad_input : tensor<128x256xbf16>
  }

  func.func private @silu_bw_rank2_decomp(
      %input: tensor<128x256xbf16>,
      %grad_output: tensor<128x256xbf16>)
      -> tensor<128x256xbf16> {
    return %input : tensor<128x256xbf16>
  }
}

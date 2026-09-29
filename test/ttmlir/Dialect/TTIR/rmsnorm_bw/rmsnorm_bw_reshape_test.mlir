// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s -o %t
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @rmsnorm_bw_rank2
  func.func @rmsnorm_bw_rank2(
      %input: tensor<128x256xbf16>,
      %gamma: tensor<256xbf16>,
      %rms: tensor<128x1xbf16>,
      %grad_output: tensor<128x256xbf16>)
      -> (tensor<128x256xbf16>, tensor<256xbf16>) {
    // CHECK-DAG: "ttir.reshape"(%arg0) {{.*}} -> tensor<1x1x128x256xbf16>
    // CHECK-DAG: "ttir.reshape"(%arg1) {{.*}} -> tensor<1x1x1x256xbf16>
    // CHECK-DAG: "ttir.reshape"(%arg2) {{.*}} -> tensor<1x1x128x1xbf16>
    // CHECK-DAG: "ttir.reshape"(%arg3) {{.*}} -> tensor<1x1x128x256xbf16>
    // CHECK: "ttcore.composite"
    // CHECK-SAME: decomposition = @rmsnorm_bw_rank2_decomp_rank4
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
    // CHECK-DAG: "ttir.reshape"({{.*}}) {{.*}} -> tensor<128x256xbf16>
    // CHECK-DAG: "ttir.reshape"({{.*}}) {{.*}} -> tensor<256xbf16>
    %grad_input, %grad_gamma = "ttcore.composite"(%input, %gamma, %rms, %grad_output) <{
        composite_name = "rmsnorm_bw",
        decomposition = @rmsnorm_bw_rank2_decomp}>
        : (tensor<128x256xbf16>, tensor<256xbf16>, tensor<128x1xbf16>,
           tensor<128x256xbf16>)
          -> (tensor<128x256xbf16>, tensor<256xbf16>)
    return %grad_input, %grad_gamma : tensor<128x256xbf16>, tensor<256xbf16>
  }

  func.func private @rmsnorm_bw_rank2_decomp(
      %input: tensor<128x256xbf16>,
      %gamma: tensor<256xbf16>,
      %rms: tensor<128x1xbf16>,
      %grad_output: tensor<128x256xbf16>)
      -> (tensor<128x256xbf16>, tensor<256xbf16>) {
    return %input, %gamma : tensor<128x256xbf16>, tensor<256xbf16>
  }

  // CHECK-LABEL: func.func @rmsnorm_bw_rank4
  func.func @rmsnorm_bw_rank4(
      %input: tensor<2x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>,
      %rms: tensor<2x1x128x1xbf16>,
      %grad_output: tensor<2x1x128x256xbf16>)
      -> (tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttcore.composite"
    // CHECK-SAME: decomposition = @rmsnorm_bw_rank4_decomp
    // CHECK-NOT: "ttir.reshape"
    // CHECK: return
    %grad_input, %grad_gamma = "ttcore.composite"(%input, %gamma, %rms, %grad_output) <{
        composite_name = "rmsnorm_bw",
        decomposition = @rmsnorm_bw_rank4_decomp}>
        : (tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>,
           tensor<2x1x128x1xbf16>, tensor<2x1x128x256xbf16>)
          -> (tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
    return %grad_input, %grad_gamma
        : tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }

  func.func private @rmsnorm_bw_rank4_decomp(
      %input: tensor<2x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>,
      %rms: tensor<2x1x128x1xbf16>,
      %grad_output: tensor<2x1x128x256xbf16>)
      -> (tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    return %input, %gamma
        : tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }
}

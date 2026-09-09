// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @layernorm_fw_rank2
  func.func @layernorm_fw_rank2(%input: tensor<128x256xbf16>, %weight: tensor<256xbf16>,
                                %bias: tensor<256xbf16>) -> tensor<128x256xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) {{.*}} -> tensor<1x1x128x256xbf16>
    // CHECK-DAG: "ttir.reshape"(%arg1) {{.*}} -> tensor<1x1x1x256xbf16>
    // CHECK-DAG: "ttir.reshape"(%arg2) {{.*}} -> tensor<1x1x1x256xbf16>
    // CHECK: "ttcore.composite"
    // CHECK-SAME: decomposition = @layernorm_fw_rank2_decomp_rank4
    // CHECK-SAME: -> tensor<1x1x128x256xbf16>
    // CHECK: "ttir.reshape"({{.*}}) {{.*}} -> tensor<128x256xbf16>
    %0 = "ttcore.composite"(%input, %weight, %bias) <{
        composite_name = "layernorm_fw",
        decomposition = @layernorm_fw_rank2_decomp,
        composite_attributes = {
          epsilon = 1.000000e-05 : f32,
          return_mean_rstd = false}}>
        : (tensor<128x256xbf16>, tensor<256xbf16>, tensor<256xbf16>)
          -> tensor<128x256xbf16>
    return %0 : tensor<128x256xbf16>
  }

  func.func private @layernorm_fw_rank2_decomp(
      %input: tensor<128x256xbf16>, %weight: tensor<256xbf16>,
      %bias: tensor<256xbf16>) -> tensor<128x256xbf16> {
    return %input : tensor<128x256xbf16>
  }

  // CHECK-LABEL: func.func @layernorm_fw_rank4
  func.func @layernorm_fw_rank4(
      %input: tensor<2x4x128x256xbf16>,
      %weight: tensor<1x1x1x256xbf16>,
      %bias: tensor<1x1x1x256xbf16>) -> tensor<2x4x128x256xbf16> {
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttcore.composite"
    // CHECK-SAME: decomposition = @layernorm_fw_rank4_decomp
    %0 = "ttcore.composite"(%input, %weight, %bias) <{
        composite_name = "layernorm_fw",
        decomposition = @layernorm_fw_rank4_decomp,
        composite_attributes = {
          epsilon = 1.000000e-05 : f32,
          return_mean_rstd = false}}>
        : (tensor<2x4x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>)
          -> tensor<2x4x128x256xbf16>
    return %0 : tensor<2x4x128x256xbf16>
  }

  func.func private @layernorm_fw_rank4_decomp(
      %input: tensor<2x4x128x256xbf16>,
      %weight: tensor<1x1x1x256xbf16>,
      %bias: tensor<1x1x1x256xbf16>) -> tensor<2x4x128x256xbf16> {
    return %input : tensor<2x4x128x256xbf16>
  }

  // CHECK-LABEL: func.func @layernorm_fw_rank3_mean_rstd
  func.func @layernorm_fw_rank3_mean_rstd(
      %input: tensor<4x128x256xbf16>, %weight: tensor<256xbf16>,
      %bias: tensor<256xbf16>)
      -> (tensor<4x128x256xbf16>, tensor<4x128x1xbf16>, tensor<4x128x1xbf16>) {
    // CHECK: "ttcore.composite"
    // CHECK-SAME: return_mean_rstd = true
    // CHECK-SAME: -> (tensor<1x4x128x256xbf16>, tensor<1x4x128x1xbf16>, tensor<1x4x128x1xbf16>)
    // CHECK-DAG: "ttir.reshape"({{.*}}) {{.*}} -> tensor<4x128x256xbf16>
    // CHECK-DAG: "ttir.reshape"({{.*}}) {{.*}} -> tensor<4x128x1xbf16>
    %0:3 = "ttcore.composite"(%input, %weight, %bias) <{
        composite_name = "layernorm_fw",
        decomposition = @layernorm_fw_rank3_decomp,
        composite_attributes = {
          epsilon = 1.000000e-05 : f32,
          return_mean_rstd = true}}>
        : (tensor<4x128x256xbf16>, tensor<256xbf16>, tensor<256xbf16>)
          -> (tensor<4x128x256xbf16>, tensor<4x128x1xbf16>, tensor<4x128x1xbf16>)
    return %0#0, %0#1, %0#2 : tensor<4x128x256xbf16>, tensor<4x128x1xbf16>, tensor<4x128x1xbf16>
  }

  func.func private @layernorm_fw_rank3_decomp(
      %input: tensor<4x128x256xbf16>, %weight: tensor<256xbf16>,
      %bias: tensor<256xbf16>)
      -> (tensor<4x128x256xbf16>, tensor<4x128x1xbf16>, tensor<4x128x1xbf16>) {
    %mean = "ttir.zeros"() <{shape = array<i32: 4, 128, 1>}> : () -> tensor<4x128x1xbf16>
    %rstd = "ttir.zeros"() <{shape = array<i32: 4, 128, 1>}> : () -> tensor<4x128x1xbf16>
    return %input, %mean, %rstd : tensor<4x128x256xbf16>, tensor<4x128x1xbf16>, tensor<4x128x1xbf16>
  }

  // CHECK-LABEL: func.func @layernorm_fw_rank5
  func.func @layernorm_fw_rank5(
      %input: tensor<2x3x4x128x256xbf16>, %weight: tensor<256xbf16>,
      %bias: tensor<256xbf16>) -> tensor<2x3x4x128x256xbf16> {
    // CHECK: "ttir.reshape"(%arg0) {{.*}} -> tensor<1x24x128x256xbf16>
    // CHECK: "ttcore.composite"
    // CHECK-SAME: -> tensor<1x24x128x256xbf16>
    // CHECK: "ttir.reshape"({{.*}}) {{.*}} -> tensor<2x3x4x128x256xbf16>
    %0 = "ttcore.composite"(%input, %weight, %bias) <{
        composite_name = "layernorm_fw",
        decomposition = @layernorm_fw_rank5_decomp,
        composite_attributes = {
          epsilon = 1.000000e-05 : f32,
          return_mean_rstd = false}}>
        : (tensor<2x3x4x128x256xbf16>, tensor<256xbf16>, tensor<256xbf16>)
          -> tensor<2x3x4x128x256xbf16>
    return %0 : tensor<2x3x4x128x256xbf16>
  }

  func.func private @layernorm_fw_rank5_decomp(
      %input: tensor<2x3x4x128x256xbf16>, %weight: tensor<256xbf16>,
      %bias: tensor<256xbf16>) -> tensor<2x3x4x128x256xbf16> {
    return %input : tensor<2x3x4x128x256xbf16>
  }
}

// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s -o %t
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @rmsnorm_fw_rank2
  func.func @rmsnorm_fw_rank2(
      %input: tensor<128x256xbf16>,
      %gamma: tensor<256xbf16>) -> tensor<128x256xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) {{.*}} -> tensor<1x1x128x256xbf16>
    // CHECK-DAG: "ttir.reshape"(%arg1) {{.*}} -> tensor<1x1x1x256xbf16>
    // CHECK: "ttcore.composite"
    // CHECK-SAME: decomposition = @rmsnorm_fw_rank2_decomp_rank4
    // CHECK-SAME: -> tensor<1x1x128x256xbf16>
    // CHECK: "ttir.reshape"({{.*}}) {{.*}} -> tensor<128x256xbf16>
    %output = "ttcore.composite"(%input, %gamma) <{
        composite_name = "rmsnorm_fw",
        decomposition = @rmsnorm_fw_rank2_decomp,
        composite_attributes = {
          return_intermediates = false,
          epsilon = 1.000000e-06 : f32}}>
        : (tensor<128x256xbf16>, tensor<256xbf16>)
          -> tensor<128x256xbf16>
    return %output : tensor<128x256xbf16>
  }

  func.func private @rmsnorm_fw_rank2_decomp(
      %input: tensor<128x256xbf16>,
      %gamma: tensor<256xbf16>) -> tensor<128x256xbf16> {
    return %input : tensor<128x256xbf16>
  }

  // CHECK-LABEL: func.func @rmsnorm_fw_rank4
  func.func @rmsnorm_fw_rank4(
      %input: tensor<2x4x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>) -> tensor<2x4x128x256xbf16> {
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttcore.composite"
    // CHECK-SAME: decomposition = @rmsnorm_fw_rank4_decomp
    // CHECK-NOT: "ttir.reshape"
    // CHECK: return
    %output = "ttcore.composite"(%input, %gamma) <{
        composite_name = "rmsnorm_fw",
        decomposition = @rmsnorm_fw_rank4_decomp,
        composite_attributes = {
          return_intermediates = false,
          epsilon = 1.000000e-06 : f32}}>
        : (tensor<2x4x128x256xbf16>, tensor<1x1x1x256xbf16>)
          -> tensor<2x4x128x256xbf16>
    return %output : tensor<2x4x128x256xbf16>
  }

  func.func private @rmsnorm_fw_rank4_decomp(
      %input: tensor<2x4x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>) -> tensor<2x4x128x256xbf16> {
    return %input : tensor<2x4x128x256xbf16>
  }

  // CHECK-LABEL: func.func @rmsnorm_fw_rank5
  func.func @rmsnorm_fw_rank5(
      %input: tensor<2x3x4x128x256xbf16>,
      %gamma: tensor<256xbf16>) -> tensor<2x3x4x128x256xbf16> {
    // CHECK: "ttir.reshape"(%arg0) {{.*}} -> tensor<1x24x128x256xbf16>
    // CHECK: "ttcore.composite"
    // CHECK-SAME: -> tensor<1x24x128x256xbf16>
    // CHECK: "ttir.reshape"({{.*}}) {{.*}} -> tensor<2x3x4x128x256xbf16>
    %output = "ttcore.composite"(%input, %gamma) <{
        composite_name = "rmsnorm_fw",
        decomposition = @rmsnorm_fw_rank5_decomp,
        composite_attributes = {
          return_intermediates = false,
          epsilon = 1.000000e-06 : f32}}>
        : (tensor<2x3x4x128x256xbf16>, tensor<256xbf16>)
          -> tensor<2x3x4x128x256xbf16>
    return %output : tensor<2x3x4x128x256xbf16>
  }

  func.func private @rmsnorm_fw_rank5_decomp(
      %input: tensor<2x3x4x128x256xbf16>,
      %gamma: tensor<256xbf16>) -> tensor<2x3x4x128x256xbf16> {
    return %input : tensor<2x3x4x128x256xbf16>
  }
}

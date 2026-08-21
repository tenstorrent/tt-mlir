// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

module {
  // Rank-2 input with rank-1 weight/bias: every operand is padded up to 4D and
  // the result is reshaped back.
  // CHECK-LABEL: func.func @layernorm_fw_rank2
  func.func @layernorm_fw_rank2(%input: tensor<128x256xbf16>, %weight: tensor<256xbf16>,
                                %bias: tensor<256xbf16>) -> tensor<128x256xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 128 : i32, 256 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 1 : i32, 256 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 1 : i32, 256 : i32]}>
    // CHECK: "ttir.layernorm_fw"
    // CHECK-SAME: -> tensor<1x1x128x256xbf16>
    // CHECK: "ttir.reshape"({{.*}}) <{shape = [128 : i32, 256 : i32]}>
    %0 = "ttir.layernorm_fw"(%input, %weight, %bias) <{
        epsilon = 1.000000e-05 : f32,
        return_mean_rstd = false}>
        : (tensor<128x256xbf16>, tensor<256xbf16>, tensor<256xbf16>)
          -> tensor<128x256xbf16>
    return %0 : tensor<128x256xbf16>
  }

  // A rank-4 op with rank-4 parameters is already legal and must be left alone.
  // CHECK-LABEL: func.func @layernorm_fw_rank4
  func.func @layernorm_fw_rank4(%input: tensor<2x4x128x256xbf16>, %weight: tensor<1x1x1x256xbf16>,
                                %bias: tensor<1x1x1x256xbf16>) -> tensor<2x4x128x256xbf16> {
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.layernorm_fw"
    %0 = "ttir.layernorm_fw"(%input, %weight, %bias) <{
        epsilon = 1.000000e-05 : f32,
        return_mean_rstd = false}>
        : (tensor<2x4x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>)
          -> tensor<2x4x128x256xbf16>
    return %0 : tensor<2x4x128x256xbf16>
  }

  // Rank-3 input returning mean/rstd: all three results are reshaped back.
  // CHECK-LABEL: func.func @layernorm_fw_rank3_mean_rstd
  func.func @layernorm_fw_rank3_mean_rstd(%input: tensor<4x128x256xbf16>, %weight: tensor<256xbf16>,
                                          %bias: tensor<256xbf16>)
      -> (tensor<4x128x256xbf16>, tensor<4x128x1xbf16>, tensor<4x128x1xbf16>) {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [1 : i32, 4 : i32, 128 : i32, 256 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 1 : i32, 256 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 1 : i32, 256 : i32]}>
    // CHECK: "ttir.layernorm_fw"
    // CHECK-SAME: return_mean_rstd = true
    // CHECK-SAME: -> (tensor<1x4x128x256xbf16>, tensor<1x4x128x1xbf16>, tensor<1x4x128x1xbf16>)
    // CHECK-DAG: "ttir.reshape"({{.*}}) <{shape = [4 : i32, 128 : i32, 256 : i32]}>
    // CHECK-DAG: "ttir.reshape"({{.*}}) <{shape = [4 : i32, 128 : i32, 1 : i32]}>
    %0, %1, %2 = "ttir.layernorm_fw"(%input, %weight, %bias) <{
        epsilon = 1.000000e-05 : f32,
        return_mean_rstd = true}>
        : (tensor<4x128x256xbf16>, tensor<256xbf16>, tensor<256xbf16>)
          -> (tensor<4x128x256xbf16>, tensor<4x128x1xbf16>, tensor<4x128x1xbf16>)
    return %0, %1, %2 : tensor<4x128x256xbf16>, tensor<4x128x1xbf16>, tensor<4x128x1xbf16>
  }

  // Rank-5 input: the leading dims are collapsed into a single dim.
  // CHECK-LABEL: func.func @layernorm_fw_rank5
  func.func @layernorm_fw_rank5(%input: tensor<2x3x4x128x256xbf16>, %weight: tensor<256xbf16>,
                                %bias: tensor<256xbf16>) -> tensor<2x3x4x128x256xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [1 : i32, 24 : i32, 128 : i32, 256 : i32]}>
    // CHECK: "ttir.layernorm_fw"
    // CHECK-SAME: -> tensor<1x24x128x256xbf16>
    // CHECK: "ttir.reshape"({{.*}}) <{shape = [2 : i32, 3 : i32, 4 : i32, 128 : i32, 256 : i32]}>
    %0 = "ttir.layernorm_fw"(%input, %weight, %bias) <{
        epsilon = 1.000000e-05 : f32,
        return_mean_rstd = false}>
        : (tensor<2x3x4x128x256xbf16>, tensor<256xbf16>, tensor<256xbf16>)
          -> tensor<2x3x4x128x256xbf16>
    return %0 : tensor<2x3x4x128x256xbf16>
  }
}

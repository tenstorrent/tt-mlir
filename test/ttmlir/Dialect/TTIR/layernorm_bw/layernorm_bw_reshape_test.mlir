// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @layernorm_bw_rank2
  func.func @layernorm_bw_rank2(
      %input: tensor<128x256xbf16>, %gamma: tensor<256xbf16>,
      %mean: tensor<128x1xbf16>, %rstd: tensor<128x1xbf16>,
      %grad: tensor<128x256xbf16>)
      -> (tensor<128x256xbf16>, tensor<256xbf16>, tensor<256xbf16>) {
    // CHECK-DAG: "ttir.reshape"(%arg0) {{.*}} -> tensor<1x1x128x256xbf16>
    // CHECK-DAG: "ttir.reshape"(%arg1) {{.*}} -> tensor<1x1x1x256xbf16>
    // CHECK-DAG: "ttir.reshape"(%arg2) {{.*}} -> tensor<1x1x128x1xbf16>
    // CHECK: "ttcore.composite"
    // CHECK-SAME: composite_name = "layernorm_bw"
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>)
    // CHECK: "ttir.reshape"({{.*}}) {{.*}} -> tensor<128x256xbf16>
    // CHECK-COUNT-2: "ttir.reshape"({{.*}}) {{.*}} -> tensor<256xbf16>
    %0:3 = "ttcore.composite"(%input, %gamma, %mean, %rstd, %grad) <{
      composite_name = "layernorm_bw",
      decomposition = @layernorm_bw_decomp,
      composite_attributes = {}
    }> : (tensor<128x256xbf16>, tensor<256xbf16>, tensor<128x1xbf16>, tensor<128x1xbf16>, tensor<128x256xbf16>) -> (tensor<128x256xbf16>, tensor<256xbf16>, tensor<256xbf16>)
    return %0#0, %0#1, %0#2 : tensor<128x256xbf16>, tensor<256xbf16>, tensor<256xbf16>
  }

  func.func private @layernorm_bw_decomp(
      %input: tensor<128x256xbf16>, %gamma: tensor<256xbf16>,
      %mean: tensor<128x1xbf16>, %rstd: tensor<128x1xbf16>,
      %grad: tensor<128x256xbf16>)
      -> (tensor<128x256xbf16>, tensor<256xbf16>, tensor<256xbf16>) {
    return %grad, %gamma, %gamma : tensor<128x256xbf16>, tensor<256xbf16>, tensor<256xbf16>
  }
}

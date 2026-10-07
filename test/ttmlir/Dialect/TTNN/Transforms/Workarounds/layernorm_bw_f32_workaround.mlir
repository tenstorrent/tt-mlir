// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --canonicalize %s | FileCheck %s

module {
  func.func public @layernorm_bw_f32(
      %input: tensor<1x1x128x256xf32>, %gamma: tensor<1x1x1x256xf32>,
      %mean: tensor<1x1x128x1xf32>, %rstd: tensor<1x1x128x1xf32>,
      %grad: tensor<1x1x128x256xf32>)
      -> (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) {
    // CHECK-LABEL: func.func public @layernorm_bw_f32
    // CHECK-COUNT-5: "ttnn.to_tensor_spec"
    // CHECK: %[[DX:.*]], %[[DGAMMA:.*]], %[[DBETA:.*]] = "ttnn.layernorm_bw"
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16{{.*}}, tensor<1x1x1x256xbf16{{.*}}, tensor<1x1x1x256xbf16
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[DX]])
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[DGAMMA]])
    // CHECK-DAG: "ttnn.to_tensor_spec"(%[[DBETA]])
    %0:3 = "ttnn.layernorm_bw"(%input, %gamma, %mean, %rstd, %grad) : (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>, tensor<1x1x128x1xf32>, tensor<1x1x128x1xf32>, tensor<1x1x128x256xf32>) -> (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>)
    return %0#0, %0#1, %0#2 : tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>
  }
}

// RUN: ttmlir-opt --ttir-to-ttnn-l1-advisor="system-desc-path=%system_desc_path% optimization-level=0" %s | FileCheck %s

module {
  func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func @add
    // CHECK: "ttnn.add"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  func.func @linear_silu(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>) -> tensor<64x256xbf16> {
    // CHECK-LABEL: func.func @linear_silu
    // CHECK: "ttnn.linear"
    // CHECK-SAME: activation = "silu"
    // CHECK-NOT: "ttnn.silu"
    %0 = "ttir.linear"(%arg0, %arg1) <{activation = "silu", transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
    return %0 : tensor<64x256xbf16>
  }
}

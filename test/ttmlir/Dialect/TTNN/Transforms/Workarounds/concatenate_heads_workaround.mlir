// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

module  {
  func.func @test_concatenate_heads_rewrite(%arg0: tensor<1x32x12x100xbf16>) -> tensor<1x12x3200xbf16>{
    // CHECK-LABEL: func.func @test_concatenate_heads_rewrite
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.reshape"
    %result = "ttnn.concatenate_heads"(%arg0) : (tensor<1x32x12x100xbf16>) -> tensor<1x12x3200xbf16>
    return %result : tensor<1x12x3200xbf16>
  }
}

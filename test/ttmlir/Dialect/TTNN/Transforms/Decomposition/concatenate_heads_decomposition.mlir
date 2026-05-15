// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Without op-model constraints (default), the decomposition pattern fires
  // unconditionally and rewrites ConcatenateHeads into Permute + Reshape.
  func.func @test_concatenate_heads_decomposition(%arg0: tensor<1x32x12x512xbf16>) -> tensor<1x12x16384xbf16> {
    // CHECK-LABEL: func.func @test_concatenate_heads_decomposition
    // CHECK-NOT: "ttnn.concatenate_heads"
    // CHECK: "ttnn.permute"
    // CHECK-NOT: "ttnn.concatenate_heads"
    // CHECK: "ttnn.reshape"
    // CHECK-NOT: "ttnn.concatenate_heads"
    %result = "ttnn.concatenate_heads"(%arg0) : (tensor<1x32x12x512xbf16>) -> tensor<1x12x16384xbf16>
    return %result : tensor<1x12x16384xbf16>
  }
}

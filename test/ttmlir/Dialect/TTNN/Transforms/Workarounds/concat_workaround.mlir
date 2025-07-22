// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func public @test_concat_datatype_workaround(%arg0: tensor<2x53xsi32>, %arg1: tensor<2x1xsi32>) -> tensor<2x54xsi32> {
    %0 = ttir.empty() : tensor<2x54xsi32>
    // CHECK-LABEL: func.func public @test_concat_datatype_workaround
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<2x53xsi32
    // CHECK-SAME: -> tensor<2x53xbf16
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<2x1xsi32
    // CHECK-SAME: -> tensor<2x1xbf16
    // CHECK: %[[CONCAT:[0-9]+]] = "ttnn.concat"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: tensor<2x53xbf16
    // CHECK-SAME: tensor<2x1xbf16
    // CHECK-SAME: -> tensor<2x54xbf16
    %1 = "ttir.concat"(%arg0, %arg1, %0) <{dim = 1 : si32}> : (tensor<2x53xsi32>, tensor<2x1xsi32>, tensor<2x54xsi32>) -> tensor<2x54xsi32>
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[CONCAT]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<2x54xbf16
    // CHECK-SAME: -> tensor<2x54xsi32
    return %1 : tensor<2x54xsi32>
  }
}

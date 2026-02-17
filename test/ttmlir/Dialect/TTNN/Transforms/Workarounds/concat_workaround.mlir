// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
// UNSUPPORTED: true

module {
  func.func public @test_concat_datatype_workaround(%arg0: tensor<2x53xsi32>, %arg1: tensor<2x1xsi32>) -> tensor<2x54xsi32> {
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
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 1 : si32}> : (tensor<2x53xsi32>, tensor<2x1xsi32>) -> tensor<2x54xsi32>
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[CONCAT]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<2x54xbf16
    // CHECK-SAME: -> tensor<2x54xsi32
    return %0 : tensor<2x54xsi32>
  }

  func.func public @test_concat_reshape_workaround(%arg0: tensor<53xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<55xf32> {
    // CHECK-LABEL: @test_concat_reshape_workaround
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: <{shape = [53 : i32, 1 : i32]}>
    // CHECK-SAME: tensor<53xf32,
    // CHECK-SAME: -> tensor<53x1xf32,
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.reshape"(%arg1)
    // CHECK-SAME: <{shape = [1 : i32, 1 : i32]}>
    // CHECK-SAME: tensor<1xf32,
    // CHECK-SAME: -> tensor<1x1xf32,
    // CHECK: %[[ARG2:[0-9]+]] = "ttnn.reshape"(%arg2)
    // CHECK-SAME: <{shape = [1 : i32, 1 : i32]}>
    // CHECK-SAME: tensor<1xf32,
    // CHECK-SAME: -> tensor<1x1xf32,
    // CHECK: %[[CONCAT:[0-9]+]] = "ttnn.concat"(%[[ARG0]], %[[ARG1]], %[[ARG2]])
    // CHECK-SAME: {dim = 0 : si32}
    // CHECK-SAME: tensor<53x1xf32,
    // CHECK-SAME: tensor<1x1xf32,
    // CHECK-SAME: tensor<1x1xf32,
    // CHECK-SAME: -> tensor<55x1xf32,
    %0 = "ttir.concat"(%arg0, %arg1, %arg2) <{dim = 0 : si32}> : (tensor<53xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<55xf32>
    // CHECK: %{{[0-9]+}} = "ttnn.reshape"(%[[CONCAT]])
    // CHECK-SAME: {shape = [55 : i32]}
    // CHECK-SAME: tensor<55x1xf32,
    // CHECK-SAME: -> tensor<55xf32,
    return %0 : tensor<55xf32>
  }
}

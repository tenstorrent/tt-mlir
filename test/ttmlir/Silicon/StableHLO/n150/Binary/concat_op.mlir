// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_concat attributes {} {
  func.func public @test_concat_0(%arg0: tensor<32x32xf32>, %arg1: tensor<64x32xf32>) -> tensor<96x32xf32> {
    // CHECK-LABEL: func.func public @test_concat_0
    // CHECK: ttnn.concat
    // CHECK-SAME: dim = 0
    // CHECK-SAME: tensor<32x32xf32,
    // CHECK-SAME: tensor<64x32xf32,
    // CHECK-SAME: -> tensor<96x32xf32,
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
    dimension = 0 : i64
    } : (tensor<32x32xf32>, tensor<64x32xf32>) -> tensor<96x32xf32>
    return %0 : tensor<96x32xf32>
  }

  func.func public @test_concat_1(%arg0: tensor<32x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<32x96xf32> {
    // CHECK-LABEL: func.func public @test_concat_1
    // CHECK: ttnn.concat
    // CHECK-SAME: dim = 1
    // CHECK-SAME: tensor<32x32xf32,
    // CHECK-SAME: tensor<32x64xf32,
    // CHECK-SAME: -> tensor<32x96xf32,
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
    dimension = 1 : i64
    } : (tensor<32x32xf32>, tensor<32x64xf32>) -> tensor<32x96xf32>
    return %0 : tensor<32x96xf32>
  }


  func.func public @test_concat_2(%arg0: tensor<128x64xf32>, %arg1: tensor<128x96xf32>) -> tensor<128x160xf32> {
    // CHECK-LABEL: func.func public @test_concat_2
    // CHECK: ttnn.concat
    // CHECK-SAME: dim = 1
    // CHECK-SAME: tensor<128x64xf32,
    // CHECK-SAME: tensor<128x96xf32,
    // CHECK-SAME: -> tensor<128x160xf32,
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 1 : i64
    } : (tensor<128x64xf32>, tensor<128x96xf32>) -> tensor<128x160xf32>
    return %0 : tensor<128x160xf32>
  }

  func.func public @test_concat_3(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x96xf32> {
    // CHECK-LABEL: func.func public @test_concat_3
    // CHECK: ttnn.concat
    // CHECK-SAME: dim = 1
    // CHECK-SAME: tensor<64x32xf32,
    // CHECK-SAME: tensor<64x64xf32,
    // CHECK-SAME: -> tensor<64x96xf32,
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 1 : i64
    } : (tensor<64x32xf32>, tensor<64x64xf32>) -> tensor<64x96xf32>
    return %0 : tensor<64x96xf32>
  }

  func.func public @test_concat_4(%arg0: tensor<32x32x32x32xf32>, %arg1: tensor<32x32x32x64xf32>) -> tensor<32x32x32x96xf32> {
    // CHECK-LABEL: func.func public @test_concat_4
    // CHECK: ttnn.concat
    // CHECK-SAME: dim = 3
    // CHECK-SAME: tensor<32x32x32x32xf32,
    // CHECK-SAME: tensor<32x32x32x64xf32,
    // CHECK-SAME: -> tensor<32x32x32x96xf32,
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 3 : i64
    } : (tensor<32x32x32x32xf32>, tensor<32x32x32x64xf32>) -> tensor<32x32x32x96xf32>
    return %0 : tensor<32x32x32x96xf32>
  }

  func.func public @test_concat_integer(%arg0: tensor<2x53xi64>, %arg1: tensor<2x1xi64>) -> tensor<2x54xi64> {
    // CHECK-LABEL: func.func public @test_concat_integer
    // CHECK: "ttnn.concat"(%arg0, %arg1)
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: tensor<2x53xsi32
    // CHECK-SAME: tensor<2x1xsi32
    // CHECK-SAME: -> tensor<2x54xsi32
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<2x53xi64>, tensor<2x1xi64>) -> tensor<2x54xi64>
    return %0 : tensor<2x54xi64>
  }

  func.func public @test_concat_reshape_workaround(%arg0: tensor<1x53xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xf32>) -> tensor<1x55xf32> {
    // CHECK-LABEL: @test_concat_reshape_workaround
    // CHECK: %{{[0-9]+}} = "ttnn.concat"(%arg0, %arg1, %arg2)
    // CHECK-SAME: {dim = 1 : si32}
    // CHECK-SAME: tensor<1x53xf32,
    // CHECK-SAME: tensor<1x1xf32,
    // CHECK-SAME: tensor<1x1xf32,
    // CHECK-SAME: -> tensor<1x55xf32,
    %0 = stablehlo.concatenate %arg0, %arg1, %arg2, dim = 1 : (tensor<1x53xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x55xf32>
    return %0 : tensor<1x55xf32>
  }
}

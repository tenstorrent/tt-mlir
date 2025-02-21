// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module @jit_eltwise_select attributes {} {
  func.func public @test_select(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK-LABEL: func.func public @test_select
    // CHECK: tensor.empty
    // CHECK: [[EQ:{{0-9}}+]] = "ttnn.eq"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: -> tensor<64x128xbf16
    %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xi1>
    // CHECK: ttnn.where
    // CHECK-SAME: [[EQ]]
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: -> tensor<64x128xbf16
    %1 = stablehlo.select %0, %arg0, %arg1 : (tensor<64x128xi1>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }

  func.func public @test_select_bf16(%arg0: tensor<1x1x1x50xi1>, %arg1: tensor<1x12x1x50xbf16>, %arg2: tensor<bf16>) -> tensor<1x12x1x50xbf16> {
    // CHECK-LABEL: func.func public @test_select_bf16(
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x1x50xi1>) -> tensor<1x12x1x50xi1>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2, 3] : (tensor<1x12x1x50xbf16>) -> tensor<1x12x1x50xbf16>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<bf16>) -> tensor<1x12x1x50xbf16>
    // CHECK: "ttnn.where"(%arg0, %arg1,
    // CHECK-SAME: tensor<1x1x1x50xbf16
    // CHECK-SAME: tensor<1x1x1x50xbf16
    // CHECK-SAME: tensor<1x1x1x50xbf16
    // CHECK-SAME: -> tensor<1x1x1x50xbf16
    %3 = stablehlo.select %0, %1, %2 : tensor<1x12x1x50xi1>, tensor<1x12x1x50xbf16>
    return %3 : tensor<1x12x1x50xbf16>
  }

  func.func public @test_select_f32(%arg0: tensor<1x12x1x51xi1>, %arg1: tensor<1x12x1x51xf32>, %arg2: tensor<1x12x1x51xf32>) -> tensor<1x12x1x51xf32> {
    // CHECK-LABEL: func.func public @test_select_bf16(
    // CHECK: %[[ARGO[0-9]+]] = {{.*}}(%arg0)
    // CHECK-SAME: dtype = #tt.supportedDataTypes<f32>
    // CHECK-SAME: tensor<1x12x1x51xbf16
    // CHECK-SAME: -> tensor<1x12x1x51xf32
    // CHECK: "ttnn.where"(%[[ARG0]], %arg1, %arg2
    // CHECK-SAME: tensor<1x12x1x51xf32
    // CHECK-SAME: tensor<1x12x1x51xf32
    // CHECK-SAME: tensor<1x12x1x51xf32
    // CHECK-SAME: -> tensor<1x12x1x51xf32
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x12x1x51xi1>, tensor<1x12x1x51xf32>
    return %0 : tensor<1x12x1x51xf32>
  }
}

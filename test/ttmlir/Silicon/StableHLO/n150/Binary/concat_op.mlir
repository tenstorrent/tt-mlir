// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_concat attributes {} {
  func.func public @test_concat_0(%arg0: tensor<32x32xbf16>, %arg1: tensor<64x32xbf16>) -> tensor<96x32xbf16> {
    // CHECK-LABEL: func.func public @test_concat_0
    // CHECK: ttnn.concat
    // CHECK-SAME: dim = 0
    // CHECK-SAME: tensor<32x32xbf16,
    // CHECK-SAME: tensor<64x32xbf16,
    // CHECK-SAME: -> tensor<96x32xbf16,
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
    dimension = 0 : i64
    } : (tensor<32x32xbf16>, tensor<64x32xbf16>) -> tensor<96x32xbf16>
    return %0 : tensor<96x32xbf16>
  }

  func.func public @test_concat_1(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x64xbf16>) -> tensor<32x96xbf16> {
    // CHECK-LABEL: func.func public @test_concat_1
    // CHECK: ttnn.concat
    // CHECK-SAME: dim = 1
    // CHECK-SAME: tensor<32x32xbf16,
    // CHECK-SAME: tensor<32x64xbf16,
    // CHECK-SAME: -> tensor<32x96xbf16,
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
    dimension = 1 : i64
    } : (tensor<32x32xbf16>, tensor<32x64xbf16>) -> tensor<32x96xbf16>
    return %0 : tensor<32x96xbf16>
  }


  func.func public @test_concat_2(%arg0: tensor<128x64xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<128x160xbf16> {
    // CHECK-LABEL: func.func public @test_concat_2
    // CHECK: ttnn.concat
    // CHECK-SAME: dim = 1
    // CHECK-SAME: tensor<128x64xbf16,
    // CHECK-SAME: tensor<128x96xbf16,
    // CHECK-SAME: -> tensor<128x160xbf16,
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 1 : i64
    } : (tensor<128x64xbf16>, tensor<128x96xbf16>) -> tensor<128x160xbf16>
    return %0 : tensor<128x160xbf16>
  }

  func.func public @test_concat_3(%arg0: tensor<64x32xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<64x96xbf16> {
    // CHECK-LABEL: func.func public @test_concat_3
    // CHECK: ttnn.concat
    // CHECK-SAME: dim = 1
    // CHECK-SAME: tensor<64x32xbf16,
    // CHECK-SAME: tensor<64x64xbf16,
    // CHECK-SAME: -> tensor<64x96xbf16,
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 1 : i64
    } : (tensor<64x32xbf16>, tensor<64x64xbf16>) -> tensor<64x96xbf16>
    return %0 : tensor<64x96xbf16>
  }

  func.func public @test_concat_4(%arg0: tensor<32x32x32x32xbf16>, %arg1: tensor<32x32x32x64xbf16>) -> tensor<32x32x32x96xbf16> {
    // CHECK-LABEL: func.func public @test_concat_4
    // CHECK: ttnn.concat
    // CHECK-SAME: dim = 3
    // CHECK-SAME: tensor<32x32x32x32xbf16,
    // CHECK-SAME: tensor<32x32x32x64xbf16,
    // CHECK-SAME: -> tensor<32x32x32x96xbf16,
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 3 : i64
    } : (tensor<32x32x32x32xbf16>, tensor<32x32x32x64xbf16>) -> tensor<32x32x32x96xbf16>
    return %0 : tensor<32x32x32x96xbf16>
  }

  func.func public @test_concat_5(%arg0: tensor<1x53xi64>, %arg1: tensor<1x1xi64>) -> tensor<1x54xi64> {
    // CHECK-LABEL: func.func public @test_concat_5
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.typecast"
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<1x53xsi32
    // CHECK-SAME: -> tensor<1x53xbf16
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.typecast"
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<1x1xsi32
    // CHECK-SAME: -> tensor<1x1xbf16
    // CHECK: %[[CONCAT:[0-9]+]] = "ttnn.concat"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: tensor<1x53xbf16
    // CHECK-SAME: tensor<1x1xbf16
    // CHECK-SAME: -> tensor<1x54xbf16
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<1x53xi64>, tensor<1x1xi64>) -> tensor<1x54xi64>
    // CHECK: "ttnn.typecast"(%[[CONCAT]])
    // CHECK-SAME: dtype = #tt.supportedDataTypes<si32>
    // CHECK-SAME: tensor<1x54xbf16
    // CHECK-SAME: -> tensor<1x54xsi32
    return %0 : tensor<1x54xi64>
  }
}

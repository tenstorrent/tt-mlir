// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_concat attributes {} {
  func.func public @test_concat(%arg0: tensor<32x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<32x96xf32> {
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
    dimension = 1 : i64
    } : (tensor<32x32xf32>, tensor<32x64xf32>) -> tensor<32x96xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.concat"(%arg0, %arg1, %0) <{dim = 1 : si32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x96xf32>) -> tensor<32x96xf32>
    return %0 : tensor<32x96xf32>
  }

  func.func public @test_concat_2(%arg0: tensor<3x2xi64>, %arg1: tensor<1x2xi64>) -> tensor<4x2xi64> {
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
    dimension = 0 : i64
    } : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.concat"(%arg0, %arg1, %0) <{dim = 0 : si32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<3x2xi32>, tensor<1x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
    return %0 : tensor<4x2xi64>
  }

  func.func public @test_concat_3(%arg0: tensor<4x3xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x8xf32> {
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
    dimension = 1 : i64
    } : (tensor<4x3xf32>, tensor<4x5xf32>) -> tensor<4x8xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.concat"(%arg0, %arg1, %0) <{dim = 1 : si32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<4x3xf32>, tensor<4x5xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }

  func.func public @test_concat_4(%arg0: tensor<128x64xf32>, %arg1: tensor<128x96xf32>) -> tensor<128x160xf32> {
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 1 : i64
    } : (tensor<128x64xf32>, tensor<128x96xf32>) -> tensor<128x160xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.concat"(%arg0, %arg1, %0) <{dim = 1 : si32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<128x64xf32>, tensor<128x96xf32>, tensor<128x160xf32>) -> tensor<128x160xf32>
    return %0 : tensor<128x160xf32>
  }

  func.func public @test_concat_5(%arg0: tensor<256x512xi64>, %arg1: tensor<256x256xi64>) -> tensor<256x768xi64> {
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 1 : i64
    } : (tensor<256x512xi64>, tensor<256x256xi64>) -> tensor<256x768xi64>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.concat"(%arg0, %arg1, %0) <{dim = 1 : si32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<256x512xi32>, tensor<256x256xi32>, tensor<256x768xi32>) -> tensor<256x768xi32>
    return %0 : tensor<256x768xi64>
  }

  func.func public @test_concat_6(%arg0: tensor<64x32xf64>, %arg1: tensor<64x64xf64>) -> tensor<64x96xf64> {
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 1 : i64
    } : (tensor<64x32xf64>, tensor<64x64xf64>) -> tensor<64x96xf64>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.concat"(%arg0, %arg1, %0) <{dim = 1 : si32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<64x32xf64>, tensor<64x64xf64>, tensor<64x96xf64>) -> tensor<64x96xf64>
    return %0 : tensor<64x96xf64>
  }

  func.func public @test_concat_7(%arg0: tensor<1000x128xi32>, %arg1: tensor<500x128xi32>) -> tensor<1500x128xi32> {
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 0 : i64
    } : (tensor<1000x128xi32>, tensor<500x128xi32>) -> tensor<1500x128xi32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.concat"(%arg0, %arg1, %0) <{dim = 0 : si32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<1000x128xi32>, tensor<500x128xi32>, tensor<1500x128xi32>) -> tensor<1500x128xi32>
    return %0 : tensor<1500x128xi32>
  }

  func.func public @test_concat_8(%arg0: tensor<3x2x4x5xf64>, %arg1: tensor<3x2x4x3xf64>) -> tensor<3x2x4x8xf64> {
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 3 : i64
    } : (tensor<3x2x4x5xf64>, tensor<3x2x4x3xf64>) -> tensor<3x2x4x8xf64>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.concat"(%arg0, %arg1, %0) <{dim = 3 : si32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<3x2x4x5xf64>, tensor<3x2x4x3xf64>, tensor<3x2x4x8xf64>) -> tensor<3x2x4x8xf64>
    return %0 : tensor<3x2x4x8xf64>
  }

  func.func public @test_concat_9(%arg0: tensor<8x4x6xi32>, %arg1: tensor<8x4x2xi32>) -> tensor<8x4x8xi32> {
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {
      dimension = 2 : i64
    } : (tensor<8x4x6xi32>, tensor<8x4x2xi32>) -> tensor<8x4x8xi32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.concat"(%arg0, %arg1, %0) <{dim = 2 : si32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<8x4x6xi32>, tensor<8x4x2xi32>, tensor<8x4x8xi32>) -> tensor<8x4x8xi32>
    return %0 : tensor<8x4x8xi32>
  }
}

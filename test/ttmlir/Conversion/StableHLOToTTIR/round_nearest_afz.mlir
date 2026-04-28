// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func public @round_nearest_afz(%arg0: tensor<4x5xf32>) -> tensor<4x5xf32> {
    // CHECK-LABEL: func.func public @round_nearest_afz
    // CHECK: %[[CONST:[0-9]+]] = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<4x5xf32>}> : () -> tensor<4x5xf32>
    // CHECK: %[[ABS:[0-9]+]] = "ttir.abs"(%arg0) : (tensor<4x5xf32>) -> tensor<4x5xf32>
    // CHECK: %[[SHIFTED:[0-9]+]] = "ttir.add"(%[[ABS]], %[[CONST]]) : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
    // CHECK: %[[FLOOR:[0-9]+]] = "ttir.floor"(%[[SHIFTED]]) : (tensor<4x5xf32>) -> tensor<4x5xf32>
    // CHECK: %[[SIGN:[0-9]+]] = "ttir.sign"(%arg0) : (tensor<4x5xf32>) -> tensor<4x5xf32>
    // CHECK: %[[RESULT:[0-9]+]] = "ttir.multiply"(%[[SIGN]], %[[FLOOR]]) : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
    // CHECK: return %[[RESULT]] : tensor<4x5xf32>
    %1 = stablehlo.round_nearest_afz %arg0 : tensor<4x5xf32>
    return %1 : tensor<4x5xf32>
  }

  func.func public @round_nearest_afz_1d(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    // CHECK-LABEL: func.func public @round_nearest_afz_1d
    // CHECK: %[[CONST:[0-9]+]] = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<10xf32>}> : () -> tensor<10xf32>
    // CHECK: %[[ABS:[0-9]+]] = "ttir.abs"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
    // CHECK: %[[SHIFTED:[0-9]+]] = "ttir.add"(%[[ABS]], %[[CONST]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    // CHECK: %[[FLOOR:[0-9]+]] = "ttir.floor"(%[[SHIFTED]]) : (tensor<10xf32>) -> tensor<10xf32>
    // CHECK: %[[SIGN:[0-9]+]] = "ttir.sign"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
    // CHECK: %[[RESULT:[0-9]+]] = "ttir.multiply"(%[[SIGN]], %[[FLOOR]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    // CHECK: return %[[RESULT]] : tensor<10xf32>
    %1 = stablehlo.round_nearest_afz %arg0 : tensor<10xf32>
    return %1 : tensor<10xf32>
  }

  func.func public @round_nearest_afz_3d(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    // CHECK-LABEL: func.func public @round_nearest_afz_3d
    // CHECK: %[[CONST:[0-9]+]] = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<2x3x4xf32>}> : () -> tensor<2x3x4xf32>
    // CHECK: %[[ABS:[0-9]+]] = "ttir.abs"(%arg0) : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    // CHECK: %[[SHIFTED:[0-9]+]] = "ttir.add"(%[[ABS]], %[[CONST]]) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    // CHECK: %[[FLOOR:[0-9]+]] = "ttir.floor"(%[[SHIFTED]]) : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    // CHECK: %[[SIGN:[0-9]+]] = "ttir.sign"(%arg0) : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    // CHECK: %[[RESULT:[0-9]+]] = "ttir.multiply"(%[[SIGN]], %[[FLOOR]]) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    // CHECK: return %[[RESULT]] : tensor<2x3x4xf32>
    %1 = stablehlo.round_nearest_afz %arg0 : tensor<2x3x4xf32>
    return %1 : tensor<2x3x4xf32>
  }

  func.func public @round_nearest_afz_f16(%arg0: tensor<3x2xf16>) -> tensor<3x2xf16> {
    // CHECK-LABEL: func.func public @round_nearest_afz_f16
    // CHECK: %[[CONST:[0-9]+]] = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<3x2xf16>}> : () -> tensor<3x2xf16>
    // CHECK: %[[ABS:[0-9]+]] = "ttir.abs"(%arg0) : (tensor<3x2xf16>) -> tensor<3x2xf16>
    // CHECK: %[[SHIFTED:[0-9]+]] = "ttir.add"(%[[ABS]], %[[CONST]]) : (tensor<3x2xf16>, tensor<3x2xf16>) -> tensor<3x2xf16>
    // CHECK: %[[FLOOR:[0-9]+]] = "ttir.floor"(%[[SHIFTED]]) : (tensor<3x2xf16>) -> tensor<3x2xf16>
    // CHECK: %[[SIGN:[0-9]+]] = "ttir.sign"(%arg0) : (tensor<3x2xf16>) -> tensor<3x2xf16>
    // CHECK: %[[RESULT:[0-9]+]] = "ttir.multiply"(%[[SIGN]], %[[FLOOR]]) : (tensor<3x2xf16>, tensor<3x2xf16>) -> tensor<3x2xf16>
    // CHECK: return %[[RESULT]] : tensor<3x2xf16>
    %1 = stablehlo.round_nearest_afz %arg0 : tensor<3x2xf16>
    return %1 : tensor<3x2xf16>
  }

  func.func public @round_nearest_afz_f64(%arg0: tensor<2x4xf64>) -> tensor<2x4xf64> {
    // CHECK-LABEL: func.func public @round_nearest_afz_f64
    // CHECK: %[[CONST:[0-9]+]] = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<2x4xf64>}> : () -> tensor<2x4xf64>
    // CHECK: %[[ABS:[0-9]+]] = "ttir.abs"(%arg0) : (tensor<2x4xf64>) -> tensor<2x4xf64>
    // CHECK: %[[SHIFTED:[0-9]+]] = "ttir.add"(%[[ABS]], %[[CONST]]) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x4xf64>
    // CHECK: %[[FLOOR:[0-9]+]] = "ttir.floor"(%[[SHIFTED]]) : (tensor<2x4xf64>) -> tensor<2x4xf64>
    // CHECK: %[[SIGN:[0-9]+]] = "ttir.sign"(%arg0) : (tensor<2x4xf64>) -> tensor<2x4xf64>
    // CHECK: %[[RESULT:[0-9]+]] = "ttir.multiply"(%[[SIGN]], %[[FLOOR]]) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x4xf64>
    // CHECK: return %[[RESULT]] : tensor<2x4xf64>
    %1 = stablehlo.round_nearest_afz %arg0 : tensor<2x4xf64>
    return %1 : tensor<2x4xf64>
  }

  func.func public @round_nearest_afz_scalar(%arg0: tensor<f32>) -> tensor<f32> {
    // CHECK-LABEL: func.func public @round_nearest_afz_scalar
    // CHECK: %[[CONST:[0-9]+]] = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    // CHECK: %[[ABS:[0-9]+]] = "ttir.abs"(%arg0) : (tensor<f32>) -> tensor<f32>
    // CHECK: %[[SHIFTED:[0-9]+]] = "ttir.add"(%[[ABS]], %[[CONST]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[FLOOR:[0-9]+]] = "ttir.floor"(%[[SHIFTED]]) : (tensor<f32>) -> tensor<f32>
    // CHECK: %[[SIGN:[0-9]+]] = "ttir.sign"(%arg0) : (tensor<f32>) -> tensor<f32>
    // CHECK: %[[RESULT:[0-9]+]] = "ttir.multiply"(%[[SIGN]], %[[FLOOR]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: return %[[RESULT]] : tensor<f32>
    %1 = stablehlo.round_nearest_afz %arg0 : tensor<f32>
    return %1 : tensor<f32>
  }
}

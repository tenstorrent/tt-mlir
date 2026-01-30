// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --tt-populate-argument-types --canonicalize --ttir-fusing --ttir-quant-dequant-conversion %s | FileCheck %s

module {
  func.func @resnet_conv2d_bias_relu_pooling_pattern(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.46909785270690918>> {
    // CHECK-LABEL: func.func @resnet_conv2d_bias_relu_pooling_pattern
    // CHECK: ttir.quantize
    // CHECK: ttir.quantize
    // CHECK: ttir.reshape
    // CHECK: ttir.quantize
    // CHECK: ttir.conv2d
    // CHECK: ttir.dequantize
    // CHECK: ttir.relu
    // CHECK: ttir.quantize
    // CHECK: ttir.max_pool2d
    // CHECK: ttir.quantize
    %1 = "ttir.quantize"(%arg0) : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 0.020787402987480164>>
    %2 = "ttir.dequantize"(%1) : (tensor<1x3x224x224x!quant.uniform<i8:f32, 0.020787402987480164>>) -> tensor<1x3x224x224xf32>
    %3 = "ttir.quantize"(%arg1) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7x!quant.uniform<i8:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>
    %4 = "ttir.dequantize"(%3) : (tensor<64x3x7x7x!quant.uniform<i8:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>) -> tensor<64x3x7x7xf32>
    %5 = "ttir.conv2d"(%2, %4) <{stride = array<i32: 2, 2>, padding = array<i32: 3, 3, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %6 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %7 = "ttir.broadcast"(%6) <{broadcast_dimensions = array<i64: 1, 1, 112, 112>}> : (tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %8 = "ttir.add"(%5, %7) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %9 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x64x112x112xf32>}> : () -> tensor<1x64x112x112xf32>
    %10 = "ttir.maximum"(%8, %9) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %11 = "ttir.permute"(%10) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x64x112x112xf32>) -> tensor<1x112x112x64xf32>
    %12 = "ttir.max_pool2d"(%11) <{ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 3, 3>, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> : (tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32>
    %13 = "ttir.permute"(%12) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x56x56x64xf32>) -> tensor<1x64x56x56xf32>
    %14 = "ttir.quantize"(%13) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.46909785270690918>>
    return %14 : tensor<1x64x56x56x!quant.uniform<i8:f32, 0.46909785270690918>>
  }
}

// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="experimental-bfp8-weights=true enable-optimizer=true system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @main(%arg0: tensor<64x3x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___conv_weight"} loc("p0.1"), %arg1: tensor<8x3x224x224xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("p1.3")) -> (tensor<8x64x112x112xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = ttir.empty() : tensor<8x64x112x112xbf16>
    %1 = "ttir.convolution"(%arg1, %arg0, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 3, 3, 3, 3>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<8x3x224x224xbf16>, tensor<64x3x7x7xbf16>, tensor<8x64x112x112xbf16>) -> tensor<8x64x112x112xbf16>
    return %1 : tensor<8x64x112x112xbf16>
  }

  func.func @conv2d_bf16(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x30x30x64xbf16> {
    // CHECK-LABEL: func.func @conv2d_bf16
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK-SAME: conv2d_config = #ttnn.conv2d_config<weights_dtype = bfp_bf8
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: conv2d_config = #ttnn.conv2d_config<weights_dtype = bfp_bf8
    %0 = ttir.empty() : tensor<16x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<16x30x30x64xbf16>) -> tensor<16x30x30x64xbf16>
    return %1 : tensor<16x30x30x64xbf16>
  }

  func.func @conv2d_f32(%arg0: tensor<16x32x32x64xf32>, %arg1: tensor<64x64x3x3xf32>, %arg2: tensor<1x1x1x64xf32>) -> tensor<16x30x30x64xf32> {
    // CHECK-LABEL: func.func @conv2d_f32
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK-SAME: conv2d_config = #ttnn.conv2d_config<weights_dtype = bfp_bf8
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: conv2d_config = #ttnn.conv2d_config<weights_dtype = bfp_bf8
    %0 = ttir.empty() : tensor<16x30x30x64xf32>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<16x32x32x64xf32>, tensor<64x64x3x3xf32>, tensor<1x1x1x64xf32>, tensor<16x30x30x64xf32>) -> tensor<16x30x30x64xf32>
    return %1 : tensor<16x30x30x64xf32>
  }

  func.func @conv2d_asymmetric_padding_bf16(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<128x64x3x3xbf16>, %arg2: tensor<1x1x1x128xbf16>) -> tensor<1x33x37x128xbf16> {
    // CHECK-LABEL: func.func @conv2d_asymmetric_padding_bf16
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK-SAME: conv2d_config = #ttnn.conv2d_config<weights_dtype = bfp_bf8
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: conv2d_config = #ttnn.conv2d_config<weights_dtype = bfp_bf8
    %0 = ttir.empty() : tensor<1x33x37x128xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: 1, 3, 2, 4>,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<128x64x3x3xbf16>, tensor<1x1x1x128xbf16>, tensor<1x33x37x128xbf16>) -> tensor<1x33x37x128xbf16>
    return %1 : tensor<1x33x37x128xbf16>
  }

  func.func @conv2d_asymmetric_padding_f32(%arg0: tensor<1x16x16x32xf32>, %arg1: tensor<64x32x5x5xf32>, %arg2: tensor<1x1x1x64xf32>) -> tensor<1x15x21x64xf32> {
    // CHECK-LABEL: func.func @conv2d_asymmetric_padding_f32
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK-SAME: conv2d_config = #ttnn.conv2d_config<weights_dtype = bfp_bf8
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: conv2d_config = #ttnn.conv2d_config<weights_dtype = bfp_bf8
    %0 = ttir.empty() : tensor<1x15x21x64xf32>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: 2, 4, 1, 5>,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x16x16x32xf32>, tensor<64x32x5x5xf32>, tensor<1x1x1x64xf32>, tensor<1x15x21x64xf32>) -> tensor<1x15x21x64xf32>
    return %1 : tensor<1x15x21x64xf32>
  }
}

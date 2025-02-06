// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for conv2d operation

// Verify that the parsing fails if tensors don't have four dimensions
module attributes {} {
  func.func @conv2d_invalid_input_shape(%arg0: tensor<32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Input must be a 4D tensor
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_invalid_weight_shape(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Weight must be a 4D tensor
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_invalid_bias_shape(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Bias must be a 4D tensor
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_invalid_output_shape(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<30x30x64xbf16> {
    %0 = tensor.empty() : tensor<30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Output must be a 4D tensor
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<30x30x64xbf16>) -> tensor<30x30x64xbf16>
    return %1 : tensor<30x30x64xbf16>
  }
}

// Verify that the parsing fails if attributes are not integers or pair of integers
// -----
module attributes {} {
  func.func @conv2d_invalid_stride_shape(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Expected integer or pair of integers, got tuple of size 3 for stride
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = array<i32: 1, 2, 3>,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_invalid_padding_shape(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Expected integer, pair, or tuple of size 4, but got tuple of size 3 for padding
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: 5, 6, 7>,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_invalid_dilation_shape(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Expected integer or pair of integers, got tuple of size 3 for dilation
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = array<i32: 11, 12, 13>,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// Verify that the parsing fails if attributes have invalid values
// -----
module attributes {} {
  func.func @conv2d_invalid_stride_values(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Stride values must be greater than 0
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = array<i32: 2, -2>,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_invalid_padding_values(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Padding values must be greater or equal than 0
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: -1, 0>,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_invalid_dilation_values(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Dilation values must be greater than 0
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = array<i32: -2, -2>,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// Verify the parsing fails if number of channels are incorrect
// -----
module attributes {} {
  func.func @conv2d_input_channels_not_divisible_by_groups(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<100x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x100xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x100xbf16>
    // CHECK: error: 'ttir.conv2d' op Number of input channels from input tensor must be divisible by the number of groups. Got 64 input channels and 10 groups
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 10: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<100x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x100xbf16>) -> tensor<1x30x30x100xbf16>
    return %1 : tensor<1x30x30x100xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_output_channels_not_divisible_by_groups(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<128x64x3x3xbf16>, %arg2: tensor<1x1x1x102xbf16>) -> tensor<1x30x30x102xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x102xbf16>
    // CHECK: error: 'ttir.conv2d' op Number of output channels from output tensor must be divisible by the number of groups. Got 102 output channels and 8 groups
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 8: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<128x64x3x3xbf16>, tensor<1x1x1x102xbf16>, tensor<1x30x30x102xbf16>) -> tensor<1x30x30x102xbf16>
    return %1 : tensor<1x30x30x102xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_input_channels_missmatch_with_weight(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x128x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Number of input channels per group must match the second dimension of the weight tensor. Got 64 input channels per group and 128 in the weight tensor
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x128x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_output_channels_missmatch_with_weight(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<128x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK:  error: 'ttir.conv2d' op Number of output channels from output tensor must match the first dimension of the weight tensor. Got 64 output channels and 128 in the weight tensor
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<128x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_output_channels_missmatch_with_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x128xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Mismatch in bias tensor dimensions. Bias tensor has 128 channels, but the output tensor has 64 channels
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x128xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_input_size_smaller_than_kernel_size(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.conv2d' op Calculated padded input size per channel: (56 x 56). Kernel size: (65 x 65). Kernel size can't be greater than actual input size
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 12: i32,
              dilation = 32: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv2d_calculated_output_size_per_channel_missmatch_with_output_tensor(%arg0: tensor<1x128x256x36xbf16>, %arg1: tensor<72x6x16x32xbf16>, %arg2: tensor<1x1x1x72xbf16>) -> tensor<1x32x32x72xbf16> {
    %0 = tensor.empty() : tensor<1x32x32x72xbf16>
    // CHECK: error: 'ttir.conv2d' op  Mismatch between calculated and got output height and width. Calculated: (9 x 9). Got output tensor height and width: (32 x 32)
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 10: i32,
              padding = array<i32: 10, 12, 6, 8>,
              dilation = array<i32: 4, 6>,
              groups = 6: i32
            }> : (tensor<1x128x256x36xbf16>, tensor<72x6x16x32xbf16>, tensor<1x1x1x72xbf16>, tensor<1x32x32x72xbf16>) -> tensor<1x32x32x72xbf16>
    return %1 : tensor<1x32x32x72xbf16>
  }
}

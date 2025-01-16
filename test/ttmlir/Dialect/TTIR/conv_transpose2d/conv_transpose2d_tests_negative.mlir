// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for conv_transpose2d operation

// Verify that the parsing fails if tensors don't have four dimensions
module attributes {} {
  func.func @conv_transpose2d_invalid_input_shape(%arg0: tensor<8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Input must be a 4D tensor
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_weight_shape(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x8x8x256xbf16> {
    %0 = tensor.empty() : tensor<1x8x8x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Weight must be a 4D tensor
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x8x8x256xbf16>) -> tensor<1x8x8x256xbf16>
    return %1 : tensor<1x8x8x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_bias_shape(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<256xbf16>) -> tensor<1x8x8x256xbf16> {
    %0 = tensor.empty() : tensor<1x8x8x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Bias must be a 4D tensor
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<256xbf16>, tensor<1x8x8x256xbf16>) -> tensor<1x8x8x256xbf16>
    return %1 : tensor<1x8x8x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_output_shape(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<10x10x256xbf16> {
    %0 = tensor.empty() : tensor<10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Output must be a 4D tensor
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<10x10x256xbf16>) -> tensor<10x10x256xbf16>
    return %1 : tensor<10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_output_shape(%arg0: tensor<4x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<2x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<2x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Batch size of input and output tensors must match
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<4x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<2x10x10x256xbf16>) -> tensor<2x10x10x256xbf16>
    return %1 : tensor<2x10x10x256xbf16>
  }
}

// Verify that the parsing fails if attributes are not integers or pair of integers
// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_stride_shape(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Expected integer or pair of integers, got tuple of size 3 for stride
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = array<i32: 1, 2, 3>,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_padding_shape(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Expected integer or pair of integers, got tuple of size 3 for padding
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: 5, 6, 7>,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_output_padding_shape(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Expected integer or pair of integers, got tuple of size 3 for output padding
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = array<i32: 8, 9, 10>,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_dilation_shape(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Expected integer or pair of integers, got tuple of size 3 for dilation
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = array<i32: 11, 12, 13>,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// Verify that the parsing fails if attributes have invalid values
// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_stride_values(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Stride values must be greater than 0
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = array<i32: 2, -2>,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_padding_values(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Padding values must be greater or equal than 0
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: -1, 0>,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_output_padding_values(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Output padding values must be greater or equal than 0
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = -6: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_invalid_dilation_values(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Dilation values must be greater than 0
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = array<i32: -2, -2>,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// Verify the parsing fails if number of channels are incorrect
// -----
module attributes {} {
  func.func @conv_transpose2d_input_channels_not_divisible_by_groups(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Number of input channels from input tensor must be divisible by the number of groups. Got 256 input channels and 3 groups
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 3: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_output_channels_not_divisible_by_groups(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x350x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x350xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x350xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Number of output channels from output tensor must be divisible by the number of groups. Got 350 output channels and 4 groups.
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 4: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x350x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x350xbf16>) -> tensor<1x10x10x350xbf16>
    return %1 : tensor<1x10x10x350xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_input_channels_missmatch_with_weight(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<128x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Number of input channels from input tensor must match the first dimension of the weight tensor. Got 256 input channels and 128 in the weight tensor.
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<128x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_output_channels_missmatch_with_weight(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Number of output channels per group must match the second dimension of the weight tensor. Got 64 output channels per group and 256 in the weight tensor.
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 4: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_output_channels_missmatch_with_bias(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x128xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Mismatch in bias tensor dimensions. Bias tensor has 128 channels, but the output tensor has 256 channels.
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x128xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// Verify the parsing fails if calculated output size per channel is below zero or different from the output tensor
// -----
module attributes {} {
  func.func @conv_transpose2d_output_channels_missmatch_with_bias(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x128xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Mismatch in bias tensor dimensions. Bias tensor has 128 channels, but the output tensor has 256 channels.
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x128xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_calculated_output_size_per_channel_below_zero(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Given input size per channel: (8 x 8). Calculated output size per channel: (-2 x -4). Output size is too small
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: 6, 7>,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}

// -----
module attributes {} {
  func.func @conv_transpose2d_calculated_output_size_per_channel_missmatch_with_output_tensor(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x2x2x256xbf16> {
    %0 = tensor.empty() : tensor<1x2x2x256xbf16>
    // CHECK: error: 'ttir.conv_transpose2d' op Mismatch between expected output size per channel and got output tensor dimensions. Expected: (10 x 10), got: (2 x 2).
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x2x2x256xbf16>) -> tensor<1x2x2x256xbf16>
    return %1 : tensor<1x2x2x256xbf16>
  }
}

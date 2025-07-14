// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for conv2d operation

// Verify that the parsing fails if tensors don't have four dimensions
module {
  func.func @conv2d_invalid_input_shape(%arg0: tensor<32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Input must be a 4D tensor
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
module {
  func.func @conv2d_invalid_weight_shape(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Weight must be a 4D tensor
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
module {
  func.func @conv2d_invalid_bias_shape(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Bias must be a 4D tensor
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
module {
  func.func @conv2d_invalid_output_shape(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Output must be a 4D tensor
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x900x64xbf16>
    return %1 : tensor<1x900x64xbf16>
  }
}

// Verify that the parsing fails if attributes are not pair of integers
// -----
module {
  func.func @conv2d_invalid_kernel_size_shape(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Kernel size attribute must have two values, got: 1
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
module {
  func.func @conv2d_invalid_stride_shape(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Stride attribute must have two values, got: 3
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1, 3>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
module {
  func.func @conv2d_invalid_dilation_shape(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Dilation attribute must have two values, got: 3
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// Verify that the parsing fails if attributes have invalid values
// -----
module {
  func.func @conv2d_invalid_stride_values(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Stride attribute values must be greater than 0, got: (2, -2)
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 2, -2>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
module {
  func.func @conv2d_invalid_padding_values(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Padding attribute values must be greater than or equal to 0, got: (-1, 0)
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: -1, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
module {
  func.func @conv2d_invalid_dilation_values(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Dilation attribute values must be greater than 0, got: (-2, -2)
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: -2, -2>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
module {
  func.func @conv2d_output_channels_missmatch_with_weight_tensor(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x32x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Expected output channels attribute (32) to match the output channels in the weight tensor (64).
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 32: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x32x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
module {
  func.func @conv2d_input_channels_missmatch_with_weight_tensor(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x32xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op  Expected input channels / groups attribute (32/1) = 32 to match the number of input channels per group in the weight tensor (64).
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 32: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x32xbf16>
    return %1 : tensor<1x1x900x32xbf16>
  }
}

// -----
module {
  func.func @conv2d_kernel_height_missmatch_with_weight_tensor(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x32xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Expected kernel height attribute (6) to match the kernel height in the weight tensor (3).
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 6, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x32xbf16>
    return %1 : tensor<1x1x900x32xbf16>
  }
}

// -----
module {
  func.func @conv2d_kernel_width_missmatch_with_weight_tensor(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x32xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Expected kernel width attribute (6) to match the kernel width in the weight tensor (3).
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 6>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x32xbf16>
    return %1 : tensor<1x1x900x32xbf16>
  }
}


// -----
module {
  func.func @conv2d_flattened_input_missmatch(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op The input tensor's flattened dimension (1024) does not match the product of batch_size_attr * input_height_attr * input_width_attr (2 * 32 * 32 = 2048).
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 2: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
module {
  func.func @conv2d_flattened_output_missmatch(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op The output tensor's flattened dimension (900) does not match the product of batch_size * output_height * output_width (2 * 14 * 30 = 840)
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 2: i32,
              input_height = 16: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
module {
  func.func @conv2d_input_channels_not_divisible_by_group(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<96x64x3x3xbf16>, %arg2: tensor<1x1x1x96xbf16>) -> tensor<1x1x900x96xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op Expected input channels / groups attribute (64/3) = 21 to match the number of input channels per group in the weight tensor (64).
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 96: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 3: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x1x1024x64xbf16>, tensor<96x64x3x3xbf16>, tensor<1x1x1x96xbf16>, !ttnn.device) -> tensor<1x1x900x96xbf16>
    return %1 : tensor<1x1x900x96xbf16>
  }
}

// -----
module {
  func.func @conv2d_kernel_size_bigger_than_input_size(%arg0: tensor<4x1x1024x64xbf16>, %arg1: tensor<64x32x12x12xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<4x1x900x64xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv2d' op The effective kernel size (67, 133) cannot be greater than the padded input size per channel (36, 40).
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64: i32,
              out_channels = 64: i32,
              batch_size = 1: i32,
              input_height = 32: i32,
              input_width = 32: i32,
              kernel_size = array<i32: 12, 12>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 2, 4>,
              dilation = array<i32: 6, 12>,
              groups = 2: i32,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<4x1x1024x64xbf16>, tensor<64x32x12x12xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<4x1x900x64xbf16>
    return %1 : tensor<4x1x900x64xbf16>
  }
}

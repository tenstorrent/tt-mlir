// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for conv3d operation

// Verify that the parsing fails if tensors don't have correct dimensions
module {
  func.func @conv3d_invalid_input_shape(%arg0: tensor<8x28x28x4xbf16>, %arg1: tensor<108x16xbf16>, %arg2: tensor<32x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op input must be a 5D tensor [N, D, H, W, C]
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<8x28x28x4xbf16>, tensor<108x16xbf16>, tensor<32x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// -----
module {
  func.func @conv3d_invalid_weight_shape(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<108x3x16xbf16>, %arg2: tensor<32x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op weight must be a 2D tensor [kD*kH*kW*C, O]
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x4xbf16>, tensor<108x3x16xbf16>, tensor<32x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// -----
module {
  func.func @conv3d_invalid_bias_shape(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<108x16xbf16>, %arg2: tensor<1x1x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op bias must be a 2D tensor [32, O]
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x4xbf16>, tensor<108x16xbf16>, tensor<1x1x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// -----
module {
  func.func @conv3d_invalid_output_shape(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<108x16xbf16>, %arg2: tensor<32x16xbf16>) -> tensor<6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op result must be a 5D tensor [N, D_out, H_out, W_out, O]
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x4xbf16>, tensor<108x16xbf16>, tensor<32x16xbf16>, !ttnn.device) -> tensor<6x26x26x16xbf16>
    return %1 : tensor<6x26x26x16xbf16>
  }
}

// Verify that the parsing fails if attributes are not triplets of integers
// -----
module {
  func.func @conv3d_invalid_kernel_size_shape(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<36x16xbf16>, %arg2: tensor<32x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op kernel_size must have 3 values, got: 2
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x4xbf16>, tensor<36x16xbf16>, tensor<32x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// -----
module {
  func.func @conv3d_invalid_stride_shape(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<108x16xbf16>, %arg2: tensor<32x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op stride must have 3 values, got: 4
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1, 2>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x4xbf16>, tensor<108x16xbf16>, tensor<32x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// -----
module {
  func.func @conv3d_invalid_padding_shape(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<108x16xbf16>, %arg2: tensor<32x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op padding must have 3 values, got: 2
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x4xbf16>, tensor<108x16xbf16>, tensor<32x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// Verify that the parsing fails if attributes have invalid values
// -----
module {
  func.func @conv3d_invalid_stride_values(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<108x16xbf16>, %arg2: tensor<32x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op stride values must be > 0, got: (1, -1, 1)
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, -1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x4xbf16>, tensor<108x16xbf16>, tensor<32x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// -----
module {
  func.func @conv3d_invalid_padding_values(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<108x16xbf16>, %arg2: tensor<32x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op padding values must be >= 0, got: (-1, 0, 1)
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: -1, 0, 1>,
              padding_mode = "zeros",
              groups = 1: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x4xbf16>, tensor<108x16xbf16>, tensor<32x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// -----
module {
  func.func @conv3d_invalid_groups(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<108x16xbf16>, %arg2: tensor<32x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op groups must be >= 1, got: 0
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 0: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x4xbf16>, tensor<108x16xbf16>, tensor<32x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// Verify that the parsing fails if tensor dimensions don't match parameters
// -----
module {
  func.func @conv3d_bias_shape_invalid(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<108x16xbf16>, %arg2: tensor<16x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op bias first dimension must be 32 (tile height), got 16
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x4xbf16>, tensor<108x16xbf16>, tensor<16x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// -----
module {
  func.func @conv3d_input_channels_not_divisible_by_groups(%arg0: tensor<1x8x28x28x7xbf16>, %arg1: tensor<54x16xbf16>, %arg2: tensor<32x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op in_channels (7) must be divisible by groups (4)
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 7: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 4: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x7xbf16>, tensor<54x16xbf16>, tensor<32x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// -----
module {
  func.func @conv3d_out_channels_not_divisible_by_groups(%arg0: tensor<1x8x28x28x8xbf16>, %arg1: tensor<54x15xbf16>, %arg2: tensor<32x15xbf16>) -> tensor<1x6x26x26x15xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op out_channels (15) must be divisible by groups (4)
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 8: i32,
              out_channels = 15: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 4: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x8xbf16>, tensor<54x15xbf16>, tensor<32x15xbf16>, !ttnn.device) -> tensor<1x6x26x26x15xbf16>
    return %1 : tensor<1x6x26x26x15xbf16>
  }
}

// -----
module {
  func.func @conv3d_weight_flattened_dim_mismatch(%arg0: tensor<1x8x28x28x8xbf16>, %arg1: tensor<216x16xbf16>, %arg2: tensor<32x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op weight flattened dimension (216) must equal kD*kH*kW*C_in/groups (108)
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 8: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 2: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x8xbf16>, tensor<216x16xbf16>, tensor<32x16xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// -----
module {
  func.func @conv3d_bias_channels_mismatch(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<108x16xbf16>, %arg2: tensor<32x32xbf16>) -> tensor<1x6x26x26x16xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.conv3d' op bias output channels (32) must match weight output channels (16)
    %1 = "ttnn.conv3d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 4: i32,
              out_channels = 16: i32,
              batch_size = 1: i32,
              input_depth = 8: i32,
              input_height = 28: i32,
              input_width = 28: i32,
              kernel_size = array<i32: 3, 3, 3>,
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<1x8x28x28x4xbf16>, tensor<108x16xbf16>, tensor<32x32xbf16>, !ttnn.device) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

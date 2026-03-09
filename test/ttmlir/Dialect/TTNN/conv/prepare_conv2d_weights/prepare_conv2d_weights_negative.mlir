// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for prepare_conv2d_weights operation.

// Verify that the parsing fails if tensor don't have four dimensions
module {
  func.func @prepare_conv2d_weights_weight_shape(%arg0: tensor<64x64x3xbf16>) -> tensor<64x64x3xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.prepare_conv2d_weights' op Weight must be a 4D tensor
    %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
            <{
              batch_size = 1 : i32,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              has_bias = true,
              in_channels = 64 : i32,
              input_height = 32 : i32,
              input_memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>,
              input_tensor_layout = #ttnn.layout<tile>,
              input_width = 32 : i32,
              kernel_size = array<i32: 3, 3>,
              out_channels = 64 : i32,
              padding = array<i32: 0, 0>,
              stride = array<i32: 1, 1>,
              weights_format = "OIHW",
              input_dtype = #ttcore.supportedDataTypes<bf16>,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<64x64x3xbf16>, !ttnn.device) -> tensor<64x64x3xbf16>
    return %1 : tensor<64x64x3xbf16>
  }
}

// Verify that the parsing fails if weight format is different than `OIHW`
// -----
module {
  func.func @prepare_conv2d_weights_invalid_weights_format(%arg0: tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.prepare_conv2d_weights' op Only `OIHW` weights format is currently supported
    %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
            <{
              batch_size = 1 : i32,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              has_bias = true,
              in_channels = 64 : i32,
              input_height = 32 : i32,
              input_memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>,
              input_tensor_layout = #ttnn.layout<tile>,
              input_width = 32 : i32,
              kernel_size = array<i32: 3, 3>,
              out_channels = 64 : i32,
              padding = array<i32: 0, 0>,
              stride = array<i32: 1, 1>,
              weights_format = "OHWI",
              input_dtype = #ttcore.supportedDataTypes<bf16>,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<64x64x3x3xbf16>
    return %1 : tensor<64x64x3x3xbf16>
  }
}

// Verify that the parsing fails if there is a mismatch between passed attributes and the weight tensor shape
// -----
module {
  func.func @prepare_conv2d_weights_out_channels_missmatch(%arg0: tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.prepare_conv2d_weights' op Expected output channels attribute (128) to match the first dimension of the weight tensor (64)
    %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
            <{
              batch_size = 1 : i32,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              has_bias = true,
              in_channels = 64 : i32,
              input_height = 32 : i32,
              input_memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>,
              input_tensor_layout = #ttnn.layout<tile>,
              input_width = 32 : i32,
              kernel_size = array<i32: 3, 3>,
              out_channels = 128 : i32,
              padding = array<i32: 0, 0>,
              stride = array<i32: 1, 1>,
              weights_format = "OIHW",
              input_dtype = #ttcore.supportedDataTypes<bf16>,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<64x64x3x3xbf16>
    return %1 : tensor<64x64x3x3xbf16>
  }
}

// -----
module {
  func.func @prepare_conv2d_weights_in_channels_missmatch(%arg0: tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.prepare_conv2d_weights' op Expected input channels attribute (128) to match the number of input channels per group (64)
    %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
            <{
              batch_size = 1 : i32,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              has_bias = true,
              in_channels = 128 : i32,
              input_height = 32 : i32,
              input_memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>,
              input_tensor_layout = #ttnn.layout<tile>,
              input_width = 32 : i32,
              kernel_size = array<i32: 3, 3>,
              out_channels = 64 : i32,
              padding = array<i32: 0, 0>,
              stride = array<i32: 1, 1>,
              weights_format = "OIHW",
              input_dtype = #ttcore.supportedDataTypes<bf16>,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<64x64x3x3xbf16>
    return %1 : tensor<64x64x3x3xbf16>
  }
}

// -----
module {
  func.func @prepare_conv2d_weights_kernel_size_invalid_height(%arg0: tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.prepare_conv2d_weights' op Expected kernel height attribute (6) to match the third dimension of the weight tensor (3)
    %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
            <{
              batch_size = 1 : i32,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              has_bias = true,
              in_channels = 64 : i32,
              input_height = 32 : i32,
              input_memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>,
              input_tensor_layout = #ttnn.layout<tile>,
              input_width = 32 : i32,
              kernel_size = array<i32: 6, 3>,
              out_channels = 64 : i32,
              padding = array<i32: 0, 0>,
              stride = array<i32: 1, 1>,
              weights_format = "OIHW",
              input_dtype = #ttcore.supportedDataTypes<bf16>,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<64x64x3x3xbf16>
    return %1 : tensor<64x64x3x3xbf16>
  }
}

// -----
module {
  func.func @prepare_conv2d_weights_kernel_size_invalid_width(%arg0: tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.prepare_conv2d_weights' op Expected kernel width attribute (7) to match the fourth dimension of the weight tensor (3)
    %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
            <{
              batch_size = 1 : i32,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              has_bias = true,
              in_channels = 64 : i32,
              input_height = 32 : i32,
              input_memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>,
              input_tensor_layout = #ttnn.layout<tile>,
              input_width = 32 : i32,
              kernel_size = array<i32: 3, 7>,
              out_channels = 64 : i32,
              padding = array<i32: 0, 0>,
              stride = array<i32: 1, 1>,
              weights_format = "OIHW",
              input_dtype = #ttcore.supportedDataTypes<bf16>,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<64x64x3x3xbf16>
    return %1 : tensor<64x64x3x3xbf16>
  }
}

// Verify that the parsing fails if attributes have invalid shape
// -----
module {
  func.func @prepare_conv2d_weights_kernel_size_invalid_shape(%arg0: tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.prepare_conv2d_weights' op Expected kernel size attribute to be a 2D tensor
    %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
            <{
              batch_size = 1 : i32,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              has_bias = true,
              in_channels = 64 : i32,
              input_height = 32 : i32,
              input_memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>,
              input_tensor_layout = #ttnn.layout<tile>,
              input_width = 32 : i32,
              kernel_size = array<i32: 3, 3, 1>,
              out_channels = 64 : i32,
              padding = array<i32: 0, 0>,
              stride = array<i32: 1, 1>,
              weights_format = "OIHW",
              input_dtype = #ttcore.supportedDataTypes<bf16>,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<64x64x3x3xbf16>
    return %1 : tensor<64x64x3x3xbf16>
  }
}

// -----
module {
  func.func @prepare_conv2d_weights_stride_invalid_shape(%arg0: tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.prepare_conv2d_weights' op Expected stride attribute to be a 2D tensor
    %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
            <{
              batch_size = 1 : i32,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              has_bias = true,
              in_channels = 64 : i32,
              input_height = 32 : i32,
              input_memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>,
              input_tensor_layout = #ttnn.layout<tile>,
              input_width = 32 : i32,
              kernel_size = array<i32: 3, 3>,
              out_channels = 64 : i32,
              padding = array<i32: 0, 0>,
              stride = array<i32: 1, 1, 1>,
              weights_format = "OIHW",
              input_dtype = #ttcore.supportedDataTypes<bf16>,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<64x64x3x3xbf16>
    return %1 : tensor<64x64x3x3xbf16>
  }
}

// -----
module {
  func.func @prepare_conv2d_weights_dilation_invalid_shape(%arg0: tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.prepare_conv2d_weights' op Expected dilation attribute to be a 2D tensor
    %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
            <{
              batch_size = 1 : i32,
              dilation = array<i32: 1, 1, 1>,
              groups = 1 : i32,
              has_bias = true,
              in_channels = 64 : i32,
              input_height = 32 : i32,
              input_memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>,
              input_tensor_layout = #ttnn.layout<tile>,
              input_width = 32 : i32,
              kernel_size = array<i32: 3, 3>,
              out_channels = 64 : i32,
              padding = array<i32: 0, 0>,
              stride = array<i32: 1, 1>,
              weights_format = "OIHW",
              input_dtype = #ttcore.supportedDataTypes<bf16>,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<64x64x3x3xbf16>
    return %1 : tensor<64x64x3x3xbf16>
  }
}

// -----
module {
  func.func @prepare_conv2d_weights_padding_invalid_shape(%arg0: tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xbf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.prepare_conv2d_weights' op Expected padding attribute to be a 2D tensor
    %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
            <{
              batch_size = 1 : i32,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              has_bias = true,
              in_channels = 64 : i32,
              input_height = 32 : i32,
              input_memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>,
              input_tensor_layout = #ttnn.layout<tile>,
              input_width = 32 : i32,
              kernel_size = array<i32: 3, 3>,
              out_channels = 64 : i32,
              padding = array<i32: 0, 0, 1>,
              stride = array<i32: 1, 1>,
              weights_format = "OIHW",
              input_dtype = #ttcore.supportedDataTypes<bf16>,
              output_dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<64x64x3x3xbf16>
    return %1 : tensor<64x64x3x3xbf16>
  }
}

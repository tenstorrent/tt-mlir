// RUN: not ttmlir-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: 'ttnn.silu_bw' op input must be a 4D tensor, got rank 3
module {
  func.func @input_not_4d(%input: tensor<1x128x256xbf16>, %grad_output: tensor<1x128x256xbf16>)
      -> tensor<1x128x256xbf16> {
    %0 = "ttnn.silu_bw"(%input, %grad_output)
        : (tensor<1x128x256xbf16>, tensor<1x128x256xbf16>)
          -> tensor<1x128x256xbf16>
    return %0 : tensor<1x128x256xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.silu_bw' op grad_output shape must match input shape, expected 1, 1, 128, 256, got 1, 1, 64, 256
module {
  func.func @grad_output_shape(%input: tensor<1x1x128x256xbf16>, %grad_output: tensor<1x1x64x256xbf16>)
      -> tensor<1x1x128x256xbf16> {
    %0 = "ttnn.silu_bw"(%input, %grad_output)
        : (tensor<1x1x128x256xbf16>, tensor<1x1x64x256xbf16>)
          -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.silu_bw' op result shape must match input shape, expected 1, 1, 128, 256, got 1, 1, 128, 1
module {
  func.func @grad_input_shape(%input: tensor<1x1x128x256xbf16>, %grad_output: tensor<1x1x128x256xbf16>)
      -> tensor<1x1x128x1xbf16> {
    %0 = "ttnn.silu_bw"(%input, %grad_output)
        : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
          -> tensor<1x1x128x1xbf16>
    return %0 : tensor<1x1x128x1xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.silu_bw' op grad_output element type must match input element type, expected 'bf16', got 'f32'
module {
  func.func @grad_output_element_type(%input: tensor<1x1x128x256xbf16>, %grad_output: tensor<1x1x128x256xf32>)
      -> tensor<1x1x128x256xbf16> {
    %0 = "ttnn.silu_bw"(%input, %grad_output)
        : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xf32>)
          -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.silu_bw' op result element type must match input element type, expected 'bf16', got 'f32'
module {
  func.func @grad_input_element_type(%input: tensor<1x1x128x256xbf16>, %grad_output: tensor<1x1x128x256xbf16>)
      -> tensor<1x1x128x256xf32> {
    %0 = "ttnn.silu_bw"(%input, %grad_output)
        : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
          -> tensor<1x1x128x256xf32>
    return %0 : tensor<1x1x128x256xf32>
  }
}

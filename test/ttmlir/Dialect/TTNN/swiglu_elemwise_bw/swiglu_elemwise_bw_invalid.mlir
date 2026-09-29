// RUN: not ttmlir-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: 'ttnn.swiglu_elemwise_bw' op input must be a 4D tensor, got rank 3
module {
  func.func @input_not_4d(%input: tensor<1x128x256xbf16>, %gate: tensor<1x128x256xbf16>, %grad_output: tensor<1x128x256xbf16>)
      -> (tensor<1x128x256xbf16>, tensor<1x128x256xbf16>) {
    %0, %1 = "ttnn.swiglu_elemwise_bw"(%input, %gate, %grad_output)
        : (tensor<1x128x256xbf16>, tensor<1x128x256xbf16>, tensor<1x128x256xbf16>)
          -> (tensor<1x128x256xbf16>, tensor<1x128x256xbf16>)
    return %0, %1 : tensor<1x128x256xbf16>, tensor<1x128x256xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.swiglu_elemwise_bw' op gate shape must match input shape, expected 1, 1, 128, 256, got 1, 1, 64, 256
module {
  func.func @gate_shape(%input: tensor<1x1x128x256xbf16>, %gate: tensor<1x1x64x256xbf16>, %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) {
    %0, %1 = "ttnn.swiglu_elemwise_bw"(%input, %gate, %grad_output)
        : (tensor<1x1x128x256xbf16>, tensor<1x1x64x256xbf16>, tensor<1x1x128x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
    return %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.swiglu_elemwise_bw' op grad_output shape must match input shape
module {
  func.func @grad_output_shape(%input: tensor<1x1x128x256xbf16>, %gate: tensor<1x1x128x256xbf16>, %grad_output: tensor<1x1x128x128xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) {
    %0, %1 = "ttnn.swiglu_elemwise_bw"(%input, %gate, %grad_output)
        : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x128xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
    return %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.swiglu_elemwise_bw' op grad_input shape must match input shape
module {
  func.func @grad_input_shape(%input: tensor<1x1x128x256xbf16>, %gate: tensor<1x1x128x256xbf16>, %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>) {
    %0, %1 = "ttnn.swiglu_elemwise_bw"(%input, %gate, %grad_output)
        : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
          -> (tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>)
    return %0, %1 : tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.swiglu_elemwise_bw' op grad_gate shape must match input shape
module {
  func.func @grad_gate_shape(%input: tensor<1x1x128x256xbf16>, %gate: tensor<1x1x128x256xbf16>, %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>) {
    %0, %1 = "ttnn.swiglu_elemwise_bw"(%input, %gate, %grad_output)
        : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>)
    return %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.swiglu_elemwise_bw' op gate element type must match input element type, expected 'bf16', got 'f32'
module {
  func.func @gate_element_type(%input: tensor<1x1x128x256xbf16>, %gate: tensor<1x1x128x256xf32>, %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) {
    %0, %1 = "ttnn.swiglu_elemwise_bw"(%input, %gate, %grad_output)
        : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xf32>, tensor<1x1x128x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
    return %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}

// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

module {
  func.func @input_rank(%input: tensor<1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>, %rms: tensor<1x128x1xbf16>, %grad_output: tensor<1x128x256xbf16>) -> (tensor<1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    // expected-error @+1 {{'ttnn.rmsnorm_bw' op input must be rank 4 (B, N, S, C)}}
    %0, %1 = "ttnn.rmsnorm_bw"(%input, %gamma, %rms, %grad_output) : (tensor<1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x128x1xbf16>, tensor<1x128x256xbf16>) -> (tensor<1x128x256xbf16>, tensor<1x1x1x256xbf16>)
    return %0, %1 : tensor<1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }
}

// -----

module {
  func.func @gamma_shape(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x2x256xbf16>, %rms: tensor<1x1x128x1xbf16>, %grad_output: tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x2x256xbf16>) {
    // expected-error @+1 {{'ttnn.rmsnorm_bw' op gamma must have shape (1, 1, 1, 256)}}
    %0, %1 = "ttnn.rmsnorm_bw"(%input, %gamma, %rms, %grad_output) : (tensor<1x1x128x256xbf16>, tensor<1x1x2x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x2x256xbf16>)
    return %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x2x256xbf16>
  }
}

// -----

module {
  func.func @rms_shape(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>, %rms: tensor<1x1x64x1xbf16>, %grad_output: tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    // expected-error @+1 {{'ttnn.rmsnorm_bw' op rms must have shape (B, N, S, 1)}}
    %0, %1 = "ttnn.rmsnorm_bw"(%input, %gamma, %rms, %grad_output) : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x64x1xbf16>, tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
    return %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }
}

// -----

module {
  func.func @grad_output_shape(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>, %rms: tensor<1x1x128x1xbf16>, %grad_output: tensor<1x1x64x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    // expected-error @+1 {{'ttnn.rmsnorm_bw' op grad_output must have the same shape as input}}
    %0, %1 = "ttnn.rmsnorm_bw"(%input, %gamma, %rms, %grad_output) : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x64x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
    return %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }
}

// -----

module {
  func.func @grad_input_shape(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>, %rms: tensor<1x1x128x1xbf16>, %grad_output: tensor<1x1x128x256xbf16>) -> (tensor<1x1x64x256xbf16>, tensor<1x1x1x256xbf16>) {
    // expected-error @+1 {{'ttnn.rmsnorm_bw' op grad_input must have the same shape as input}}
    %0, %1 = "ttnn.rmsnorm_bw"(%input, %gamma, %rms, %grad_output) : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>) -> (tensor<1x1x64x256xbf16>, tensor<1x1x1x256xbf16>)
    return %0, %1 : tensor<1x1x64x256xbf16>, tensor<1x1x1x256xbf16>
  }
}

// -----

module {
  func.func @grad_gamma_shape(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>, %rms: tensor<1x1x128x1xbf16>, %grad_output: tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) {
    // expected-error @+1 {{'ttnn.rmsnorm_bw' op grad_gamma must have the same shape as gamma}}
    %0, %1 = "ttnn.rmsnorm_bw"(%input, %gamma, %rms, %grad_output) : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
    return %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}

// -----

module {
  func.func @element_types(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xf32>, %rms: tensor<1x1x128x1xbf16>, %grad_output: tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    // expected-error @+1 {{'ttnn.rmsnorm_bw' op all operands and results must have the same element type}}
    %0, %1 = "ttnn.rmsnorm_bw"(%input, %gamma, %rms, %grad_output) : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xf32>, tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
    return %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }
}

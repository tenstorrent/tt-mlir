// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

module {
  func.func @input_rank(%input: tensor<1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>) -> tensor<1x128x256xbf16> {
    // expected-error @+1 {{'ttnn.rmsnorm_fw' op input must be rank 4 (B, N, S, C)}}
    %0 = "ttnn.rmsnorm_fw"(%input, %gamma) <{return_intermediates = false}> : (tensor<1x128x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x128x256xbf16>
    return %0 : tensor<1x128x256xbf16>
  }
}

// -----

module {
  func.func @gamma_shape(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x2x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // expected-error @+1 {{'ttnn.rmsnorm_fw' op gamma must have shape (1, 1, 1, 256)}}
    %0 = "ttnn.rmsnorm_fw"(%input, %gamma) <{return_intermediates = false}> : (tensor<1x1x128x256xbf16>, tensor<1x1x2x256xbf16>) -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }
}

// -----

module {
  func.func @output_shape(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>) -> tensor<1x1x64x256xbf16> {
    // expected-error @+1 {{'ttnn.rmsnorm_fw' op output must have the same shape as input}}
    %0 = "ttnn.rmsnorm_fw"(%input, %gamma) <{return_intermediates = false}> : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x1x64x256xbf16>
    return %0 : tensor<1x1x64x256xbf16>
  }
}

// -----

module {
  func.func @element_types(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xf32>) -> tensor<1x1x128x256xbf16> {
    // expected-error @+1 {{'ttnn.rmsnorm_fw' op input, gamma and output must have the same element type}}
    %0 = "ttnn.rmsnorm_fw"(%input, %gamma) <{return_intermediates = false}> : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xf32>) -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }
}

// -----

module {
  func.func @missing_rms(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // expected-error @+1 {{'ttnn.rmsnorm_fw' op rms result must be present iff return_intermediates is true}}
    %0 = "ttnn.rmsnorm_fw"(%input, %gamma) <{return_intermediates = true}> : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }
}

// -----

module {
  func.func @rms_shape(%input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x64x1xbf16>) {
    // expected-error @+1 {{'ttnn.rmsnorm_fw' op rms must have shape (B, N, S, 1)}}
    %0, %1 = "ttnn.rmsnorm_fw"(%input, %gamma) <{return_intermediates = true}> : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x64x1xbf16>)
    return %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x64x1xbf16>
  }
}

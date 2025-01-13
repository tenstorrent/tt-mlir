// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for linear operation

// Verify that the parsing fails if either of operands is a scalar
module {
  func.func @linear_negative_1d_1d_scalar_a(%arg0: tensor<bf16>, %arg1: tensor<64xbf16>) -> tensor<1xbf16> {
    // CHECK: error: 'ttir.linear' op Input A must be at least a 1D tensor
    %0 = tensor.empty() : tensor<1xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<bf16>, tensor<64xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }
}

// -----
module {
  func.func @linear_negative_1d_1d_scalar_b(%arg0: tensor<128xbf16>, %arg1: tensor<bf16>) -> tensor<1xbf16> {
    // CHECK: error: 'ttir.linear' op Input B must be at least a 1D tensor
    %0 = tensor.empty() : tensor<1xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<128xbf16>, tensor<bf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }
}

// -----
module {
  func.func @linear_negative_1d_1d_scalar_bias(%arg0: tensor<128xbf16>, %arg1: tensor<128xbf16>, %bias: tensor<bf16>) -> tensor<1xbf16> {
    // CHECK: error: 'ttir.linear' op Bias must be at least a 1D tensor
    %0 = tensor.empty() : tensor<1xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<128xbf16>, tensor<128xbf16>, tensor<bf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }
}

// Verifty that the parsing fails if the output is a scalar
// -----
module {
  func.func @linear_negative_1d_1d_scalar_output(%arg0: tensor<128xbf16>, %arg1: tensor<128xbf16>) -> tensor<bf16> {
    // CHECK: error: 'ttir.linear' op Scalar output is not supported, output must be at least a 1D tensor
    %0 = tensor.empty() : tensor<bf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<128xbf16>, tensor<128xbf16>, tensor<bf16>) -> tensor<bf16>
    return %1 : tensor<bf16>
  }
}

// -----
module {
  func.func @linear_negative_1d_1d_output_dimension_mismatch(%arg0: tensor<128xbf16>, %arg1: tensor<128xbf16>) -> tensor<2xbf16> {
    // CHECK: error: 'ttir.linear' op Scalar output must be a 1D tensor of size 1
    %0 = tensor.empty() : tensor<2xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<128xbf16>, tensor<128xbf16>, tensor<2xbf16>) -> tensor<2xbf16>
    return %1 : tensor<2xbf16>
  }
}

// Inner dimension mismatch tests
// -----
module {
  func.func @linear_negative_1d_1d_inner_dimension_mismatch(%arg0: tensor<128xbf16>, %arg1: tensor<64xbf16>) -> tensor<1xbf16> {
    // CHECK: error: 'ttir.linear' op Input A[-1](128) and B[-2](64) must have matching inner dimensions
    %0 = tensor.empty() : tensor<1xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<128xbf16>, tensor<64xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }
}

// -----
module {
func.func @linear_negative_1d_2d_inner_dimension_mismatch(%arg0: tensor<64xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<64xbf16> {
    // CHECK: error: 'ttir.linear' op Input A[-1](64) and B[-2](128) must have matching inner dimensions
    %0 = tensor.empty() : tensor<64xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<64xbf16>, tensor<128x64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    return %1 : tensor<64xbf16>
  }
}

// -----
module {
  func.func @linear_negative_2d_1d_inner_dimension_mismatch(%arg0: tensor<64x128xbf16>, %arg1: tensor<64xbf16>) -> tensor<64xbf16> {
   // CHECK: error: 'ttir.linear' op Input A[-1](128) and B[-2](64) must have matching inner dimensions
    %0 = tensor.empty() : tensor<64xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    return %1 : tensor<64xbf16>
  }
}

// -----
module {
  func.func @linear_negative_2d_2d_inner_dimension_mismatch(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x64xbf16> {
    // CHECK: error: 'ttir.linear' op Input A[-1](128) and B[-2](64) must have matching inner dimensions
    %0 = tensor.empty() : tensor<64x64xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }
}

// -----
module {
  func.func @linear_negative_nd_nd_inner_dimension_mismatch(%arg0: tensor<7x64x128xbf16>, %arg1: tensor<1x64x128xbf16>) -> tensor<7x64x64xbf16> {
    // CHECK: error: 'ttir.linear' op Input A[-1](128) and B[-2](64) must have matching inner dimensions
    %0 = tensor.empty() : tensor<7x64x64xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<7x64x128xbf16>, tensor<1x64x128xbf16>, tensor<7x64x64xbf16>) -> tensor<7x64x64xbf16>
    return %1 : tensor<7x64x64xbf16>
  }
}

// Batch dimension mismatch tests
// -----
module {
  func.func @linear_negative_nd_nd_same_rank_batch_broadcast_incompatible_1(%arg0: tensor<7x64x128xbf16>, %arg1: tensor<2x128x64xbf16>) -> tensor<7x64x64xbf16> {
   // CHECK: error: 'ttir.linear' op Batch dimensions of input A(7) and B(2) are not broadcast compatible
    %0 = tensor.empty() : tensor<7x64x64xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<7x64x128xbf16>, tensor<2x128x64xbf16>, tensor<7x64x64xbf16>) -> tensor<7x64x64xbf16>
    return %1 : tensor<7x64x64xbf16>
  }
}

// -----
module {
  func.func @linear_negative_nd_nd_same_rank_batch_broadcast_incompatible_2(%arg0: tensor<2x7x64x128xbf16>, %arg1: tensor<7x1x128x64xbf16>) -> tensor<7x7x64x64xbf16> {
    // CHECK: error: 'ttir.linear' op Batch dimensions of input A(2,7) and B(7,1) are not broadcast compatible
    %0 = tensor.empty() : tensor<7x64x64xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<2x7x64x128xbf16>, tensor<7x1x128x64xbf16>, tensor<7x64x64xbf16>) -> tensor<7x7x64x64xbf16>
    return %1 : tensor<7x7x64x64xbf16>
  }
}

// -----
module {
  func.func @linear_negative_nd_nd_different_rank_batch_broadcast_incompatible(%arg0: tensor<12x2x7x64x128xbf16>, %arg1: tensor<7x1x128x64xbf16>) -> tensor<12x7x7x64x64xbf16> {
    // CHECK: error: 'ttir.linear' op Batch dimensions of input A(12,2,7) and B(7,1) are not broadcast compatible
    %0 = tensor.empty() : tensor<12x7x7x64x64xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<12x2x7x64x128xbf16>, tensor<7x1x128x64xbf16>, tensor<12x7x7x64x64xbf16>) -> tensor<12x7x7x64x64xbf16>
    return %1 : tensor<12x7x7x64x64xbf16>
  }
}

// Bias shape mismatch tests
// -----
module {
  func.func @linear_negative_matmul_bias_broadcast_incompatible(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<2x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: error: 'ttir.linear' op Bias shape(2,64) is not broadcast compatible with the matmul output shape(64,64)
    %0 = tensor.empty() : tensor<64x64xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<2x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }
}

// -----
module {
  func.func @linear_negative_nd_nd_matmul_bias_broadcast_incompatible(%arg0: tensor<3x64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<2x64x64xbf16>) -> tensor<3x64x64xbf16> {
    // CHECK: error: 'ttir.linear' op Bias shape(2,64,64) is not broadcast compatible with the matmul output shape(3,64,64)
    %0 = tensor.empty() : tensor<3x64x64xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<3x64x128xbf16>, tensor<128x64xbf16>, tensor<2x64x64xbf16>, tensor<3x64x64xbf16>) -> tensor<3x64x64xbf16>
    return %1 : tensor<3x64x64xbf16>
  }
}

// Output shape mismatch tests
// -----
module {
  func.func @linear_negative_2d_2d_output_shape_mismatch(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<64xbf16> {
    // CHECK: error: 'ttir.linear' op Output shape rank(1) must match the expected output shape rank(2)
    %0 = tensor.empty() : tensor<64xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    return %1 : tensor<64xbf16>
  }
}

// -----
module {
  func.func @linear_negative_2d_2d_output_shape_mismatch(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<64x128xbf16> {
    // CHECK: error: 'ttir.linear' op Output shape dimension[1](128) doesn't match the expected output shape dimension[1](64)
    %0 = tensor.empty() : tensor<64x128xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}

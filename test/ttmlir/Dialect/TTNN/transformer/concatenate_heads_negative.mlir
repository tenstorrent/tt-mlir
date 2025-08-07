// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for concatenate heads operation.

// Verify that the parsing fails if input tensor is not 4D.
module {
  func.func @concatenate_heads_invalid_1(%arg0: tensor<1x24x32xbf16>) -> tensor<1x32x3072xbf16> {
    %0 = ttir.empty() : tensor<1x32x3072xbf16>
    // CHECK: error: 'ttir.concatenate_heads' op expected rank of input tensor is 4, got rank 3
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x24x32xbf16>, tensor<1x32x3072xbf16>) -> tensor<1x32x3072xbf16>
    return %1 : tensor<1x32x3072xbf16>
  }
}

// Verify that the parsing fails if output tensor is not 3D.
module {
  func.func @concatenate_heads_invalid_2(%arg0: tensor<1x24x32x128xbf16>) -> tensor<32x3072xbf16> {
    %0 = ttir.empty() : tensor<32x3072xbf16>
    // CHECK: error: 'ttir.concatenate_heads' op expected rank of output tensor is 3, got rank 2
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x24x32x128xbf16>, tensor<32x3072xbf16>) -> tensor<32x3072xbf16>
    return %1 : tensor<32x3072xbf16>
  }
}

// Verify that the parsing fails if batch sizes do not match.
module {
  func.func @concatenate_heads_invalid_3(%arg0: tensor<1x24x32x128xbf16>) -> tensor<2x32x3072xbf16> {
    %0 = ttir.empty() : tensor<2x32x3072xbf16>
    // CHECK: error: 'ttir.concatenate_heads' op expected output batch dimension to be 1, got 2
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x24x32x128xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
    return %1 : tensor<2x32x3072xbf16>
  }
}

// Verify that the parsing fails if the sequence size does not match.
module {
  func.func @concatenate_heads_invalid_4(%arg0: tensor<1x24x32x128xbf16>) -> tensor<1x24x3072xbf16> {
    %0 = ttir.empty() : tensor<1x24x3072xbf16>
    // CHECK: error: 'ttir.concatenate_heads' op expected output sequence dimension to be 32, got 24
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x24x32x128xbf16>, tensor<1x24x3072xbf16>) -> tensor<1x24x3072xbf16>
    return %1 : tensor<1x24x3072xbf16>
  }
}

// Verify that the parsing fails if input num_heads * head_size does not match output hidden dimension.
module {
  func.func @concatenate_heads_invalid_5(%arg0: tensor<1x24x32x128xbf16>) -> tensor<1x32x2048xbf16> {
    %0 = ttir.empty() : tensor<1x32x2048xbf16>
    // CHECK: error: 'ttir.concatenate_heads' op expected output hidden dimension to be num_heads * head_size = 3072, got 2048
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x24x32x128xbf16>, tensor<1x32x2048xbf16>) -> tensor<1x32x2048xbf16>
    return %1 : tensor<1x32x2048xbf16>
  }
}

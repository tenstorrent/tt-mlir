// RUN: not ttmlir-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: 'ttnn.chunk_gated_delta_rule' op chunk_size must be a positive multiple of 32
module {
  func.func @invalid_chunk_size(
      %q: tensor<1x64x4x32xbf16>, %k: tensor<1x64x4x32xbf16>,
      %v: tensor<1x64x8x32xbf16>, %g: tensor<1x64x8xf32>,
      %beta: tensor<1x64x8xf32>) -> tensor<1x64x8x32xbf16> {
    %0 = "ttnn.chunk_gated_delta_rule"(%q, %k, %v, %g, %beta) <{
      chunk_size = 16 : ui32,
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>
    }> : (tensor<1x64x4x32xbf16>, tensor<1x64x4x32xbf16>,
          tensor<1x64x8x32xbf16>, tensor<1x64x8xf32>,
          tensor<1x64x8xf32>) -> tensor<1x64x8x32xbf16>
    return %0 : tensor<1x64x8x32xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.chunk_gated_delta_rule' op key head dimension must be a multiple of 32
module {
  func.func @invalid_key_dim(
      %q: tensor<1x64x4x16xbf16>, %k: tensor<1x64x4x16xbf16>,
      %v: tensor<1x64x8x32xbf16>, %g: tensor<1x64x8xf32>,
      %beta: tensor<1x64x8xf32>) -> tensor<1x64x8x32xbf16> {
    %0 = "ttnn.chunk_gated_delta_rule"(%q, %k, %v, %g, %beta) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>
    }> : (tensor<1x64x4x16xbf16>, tensor<1x64x4x16xbf16>,
          tensor<1x64x8x32xbf16>, tensor<1x64x8xf32>,
          tensor<1x64x8xf32>) -> tensor<1x64x8x32xbf16>
    return %0 : tensor<1x64x8x32xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.chunk_gated_delta_rule' op value head dimension must be a multiple of 32
module {
  func.func @invalid_value_dim(
      %q: tensor<1x64x4x32xbf16>, %k: tensor<1x64x4x32xbf16>,
      %v: tensor<1x64x8x16xbf16>, %g: tensor<1x64x8xf32>,
      %beta: tensor<1x64x8xf32>) -> tensor<1x64x8x16xbf16> {
    %0 = "ttnn.chunk_gated_delta_rule"(%q, %k, %v, %g, %beta) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>
    }> : (tensor<1x64x4x32xbf16>, tensor<1x64x4x32xbf16>,
          tensor<1x64x8x16xbf16>, tensor<1x64x8xf32>,
          tensor<1x64x8xf32>) -> tensor<1x64x8x16xbf16>
    return %0 : tensor<1x64x8x16xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.chunk_gated_delta_rule' op flat query/key inputs require chunk_size to be 32
module {
  func.func @invalid_flat_qk_chunk_size(
      %q: tensor<1x64x128xbf16>, %k: tensor<1x64x128xbf16>,
      %v: tensor<1x64x8x32xbf16>, %g: tensor<1x64x8xf32>,
      %beta: tensor<1x64x8xf32>) -> tensor<1x64x8x32xbf16> {
    %0 = "ttnn.chunk_gated_delta_rule"(%q, %k, %v, %g, %beta) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>
    }> : (tensor<1x64x128xbf16>, tensor<1x64x128xbf16>,
          tensor<1x64x8x32xbf16>, tensor<1x64x8xf32>,
          tensor<1x64x8xf32>) -> tensor<1x64x8x32xbf16>
    return %0 : tensor<1x64x8x32xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.chunk_gated_delta_rule' op flat Q/K/V inputs require sequence length to be divisible by chunk_size
module {
  func.func @invalid_flat_sequence_length(
      %q: tensor<1x48x4x32xbf16>, %k: tensor<1x48x4x32xbf16>,
      %v: tensor<1x48x256xbf16>, %g: tensor<1x48x8xf32>,
      %beta: tensor<1x48x8xf32>) -> tensor<1x48x8x32xbf16> {
    %0 = "ttnn.chunk_gated_delta_rule"(%q, %k, %v, %g, %beta) <{
      chunk_size = 32 : ui32,
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>
    }> : (tensor<1x48x4x32xbf16>, tensor<1x48x4x32xbf16>,
          tensor<1x48x256xbf16>, tensor<1x48x8xf32>,
          tensor<1x48x8xf32>) -> tensor<1x48x8x32xbf16>
    return %0 : tensor<1x48x8x32xbf16>
  }
}

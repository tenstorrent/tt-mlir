// RUN: ttmlir-opt %s --split-input-file --verify-diagnostics

module {
  func.func @invalid_k(
      %input: tensor<1x64x!ttcore.tile<32x32, f32>>,
      %scratch: tensor<1x64x!ttcore.tile<32x32, f32>>,
      %output: tensor<1x1x!ttcore.tile<32x32, f32>>) {
    // expected-error @+1 {{currently supports only k = 32}}
    %0 = "d2m.topk_values"(%input, %scratch, %output) <{k = 16 : i64}> : (tensor<1x64x!ttcore.tile<32x32, f32>>, tensor<1x64x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
    return
  }
}

// -----

module {
  func.func @invalid_largest(
      %input: tensor<1x64x!ttcore.tile<32x32, f32>>,
      %scratch: tensor<1x64x!ttcore.tile<32x32, f32>>,
      %output: tensor<1x1x!ttcore.tile<32x32, f32>>) {
    // expected-error @+1 {{currently supports only largest = true}}
    %0 = "d2m.topk_values"(%input, %scratch, %output) <{k = 32 : i64, largest = false}> : (tensor<1x64x!ttcore.tile<32x32, f32>>, tensor<1x64x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
    return
  }
}

// -----

module {
  func.func @invalid_stable_sort(
      %input: tensor<1x64x!ttcore.tile<32x32, f32>>,
      %scratch: tensor<1x64x!ttcore.tile<32x32, f32>>,
      %output: tensor<1x1x!ttcore.tile<32x32, f32>>) {
    // expected-error @+1 {{currently supports only stable_sort = false}}
    %0 = "d2m.topk_values"(%input, %scratch, %output) <{k = 32 : i64, stable_sort = true}> : (tensor<1x64x!ttcore.tile<32x32, f32>>, tensor<1x64x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
    return
  }
}

// -----

module {
  func.func @invalid_scratch_shape(
      %input: tensor<1x64x!ttcore.tile<32x32, f32>>,
      %scratch: tensor<1x32x!ttcore.tile<32x32, f32>>,
      %output: tensor<1x1x!ttcore.tile<32x32, f32>>) {
    // expected-error @+1 {{scratch shape must match input shape}}
    %0 = "d2m.topk_values"(%input, %scratch, %output) <{k = 32 : i64}> : (tensor<1x64x!ttcore.tile<32x32, f32>>, tensor<1x32x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
    return
  }
}

// -----

module {
  func.func @invalid_block_shape(
      %input: tensor<1x32x!ttcore.tile<32x32, f32>>,
      %scratch: tensor<1x32x!ttcore.tile<32x32, f32>>,
      %output: tensor<1x1x!ttcore.tile<32x32, f32>>) {
    // expected-error @+1 {{currently supports input block shape 1x64 tiles and output block shape 1x1 tile}}
    %0 = "d2m.topk_values"(%input, %scratch, %output) <{k = 32 : i64}> : (tensor<1x32x!ttcore.tile<32x32, f32>>, tensor<1x32x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
    return
  }
}

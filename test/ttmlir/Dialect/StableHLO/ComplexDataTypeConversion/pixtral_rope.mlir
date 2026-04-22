// RUN: ttmlir-opt --stablehlo-complex-math-expander --stablehlo-complex-data-type-conversion %s
// REQUIRES: stablehlo

// Pixtral/Mistral-Small vision encoder 2D RoPE pattern.
//
// freqs_cis is a (H x W x head_dim/2) complex position cache gathered by
// flattened (H*W) patch position indices before being applied to Q and K.
//
// This test verifies that GatherOp on a complex-typed operand is correctly
// converted by the stablehlo-complex-data-type-conversion pass:
//   - operand: tensor<110x110x32xcomplex<f32>> -> tensor<110x110x32x2xf32>
//   - slice_sizes: [1, 1, 32] -> [1, 1, 32, 2]
//   - offset_dims: [1]        -> [1, 2]
//   - result:  tensor<12100x32xcomplex<f32>>  -> tensor<12100x32x2xf32>

func.func @pixtral_rope_gather(
    %freqs_cis: tensor<110x110x32xcomplex<f32>>,
    %positions: tensor<12100x2xi64>,
    %xq: tensor<1x12100x16x64xf32>
) -> tensor<1x12100x16x64xf32> {
  // Gather freqs_cis at all 12100 patch positions -> tensor<12100x32xcomplex<f32>>
  %0 = "stablehlo.gather"(%freqs_cis, %positions) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 1, 1, 32>,
    indices_are_sorted = false
  } : (tensor<110x110x32xcomplex<f32>>, tensor<12100x2xi64>) -> tensor<12100x32xcomplex<f32>>

  // Reshape and broadcast for Q/K application: [12100x32] -> [1x12100x16x32]
  %1 = stablehlo.reshape %0 : (tensor<12100x32xcomplex<f32>>) -> tensor<1x12100x1x32xcomplex<f32>>
  %freqs_bc = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x12100x1x32xcomplex<f32>>) -> tensor<1x12100x16x32xcomplex<f32>>

  // Build complex Q from interleaved real/imag pairs
  %xq_r = stablehlo.reshape %xq : (tensor<1x12100x16x64xf32>) -> tensor<1x12100x16x32x2xf32>
  %xq_re_s = stablehlo.slice %xq_r [0:1, 0:12100, 0:16, 0:32, 0:1] : (tensor<1x12100x16x32x2xf32>) -> tensor<1x12100x16x32x1xf32>
  %xq_re = stablehlo.reshape %xq_re_s : (tensor<1x12100x16x32x1xf32>) -> tensor<1x12100x16x32xf32>
  %xq_im_s = stablehlo.slice %xq_r [0:1, 0:12100, 0:16, 0:32, 1:2] : (tensor<1x12100x16x32x2xf32>) -> tensor<1x12100x16x32x1xf32>
  %xq_im = stablehlo.reshape %xq_im_s : (tensor<1x12100x16x32x1xf32>) -> tensor<1x12100x16x32xf32>
  %xq_complex = stablehlo.complex %xq_re, %xq_im : tensor<1x12100x16x32xcomplex<f32>>

  // Apply rotation: complex multiply then extract real/imag
  %rotated = stablehlo.multiply %xq_complex, %freqs_bc : tensor<1x12100x16x32xcomplex<f32>>
  %out_re = stablehlo.real %rotated : (tensor<1x12100x16x32xcomplex<f32>>) -> tensor<1x12100x16x32xf32>
  %out_re_s = stablehlo.reshape %out_re : (tensor<1x12100x16x32xf32>) -> tensor<1x12100x16x32x1xf32>
  %out_im = stablehlo.imag %rotated : (tensor<1x12100x16x32xcomplex<f32>>) -> tensor<1x12100x16x32xf32>
  %out_im_s = stablehlo.reshape %out_im : (tensor<1x12100x16x32xf32>) -> tensor<1x12100x16x32x1xf32>
  %out_cat = stablehlo.concatenate %out_re_s, %out_im_s, dim = 4 : (tensor<1x12100x16x32x1xf32>, tensor<1x12100x16x32x1xf32>) -> tensor<1x12100x16x32x2xf32>
  %out = stablehlo.reshape %out_cat : (tensor<1x12100x16x32x2xf32>) -> tensor<1x12100x16x64xf32>
  return %out : tensor<1x12100x16x64xf32>
}

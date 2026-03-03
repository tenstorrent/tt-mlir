// RUN: ttmlir-opt %s --split-input-file --verify-diagnostics

// Test: logical_shape must have at least 2 dimensions (only 1 dimension)
// expected-error @+1 {{logical_shape must have at least 2 dimensions, got 1}}
#layout_1d = #ttcore.metal_layout<logical_shape = 32, dim_alignments = 32, collapsed_intervals = dense<[[0, 1]]> : tensor<1x2xi64>, undef, l1, sharded>

// -----

// Test: dim_alignments size must match logical_shape rank
// expected-error @+1 {{dim_alignments size (3) must match logical_shape rank (2)}}
#layout_mismatched_alignments = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// -----

// Test: alignments must be positive (zero alignment)
// expected-error @+1 {{dim_alignments[0] must be positive, got 0}}
#layout_zero_alignment = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 0x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// -----

// Note: Negative alignments (e.g., 32x-1) cannot be tested via MLIR text parsing
// because the parser rejects negative values in dimension lists before the verifier runs.
// The verifier still protects against negative alignments when attributes are constructed programmatically.

// Test: collapsed_intervals must be 2D (1D array instead)
// expected-error @+1 {{collapsed_intervals must be a 2D array, got rank 1}}
#layout_1d_intervals = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[0, 1]> : tensor<2xi64>, undef, l1, sharded>

// -----

// Test: collapsed_intervals must have pairs (second dim size 2)
// expected-error @+1 {{collapsed_intervals must have pairs (second dim size 2), got 3}}
#layout_wrong_interval_size = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1, 2]]> : tensor<1x3xi64>, undef, l1, sharded>

// -----

// Test: collapsed_intervals start index out of bounds (too large positive)
// expected-error @+1 {{collapsed_intervals start index 5 (normalized: 5) is out of bounds for logical_shape rank 2}}
#layout_start_oob = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[5, 6]]> : tensor<1x2xi64>, undef, l1, sharded>

// -----

// Test: collapsed_intervals end index out of bounds
// expected-error @+1 {{collapsed_intervals end index 10 (normalized: 10) is out of bounds for logical_shape rank 2}}
#layout_end_oob = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 10]]> : tensor<1x2xi64>, undef, l1, sharded>

// -----

// Test: collapsed_intervals start must not exceed end (inverted range)
// expected-error @+1 {{collapsed_intervals start (1) must not exceed end (0)}}
#layout_inverted_interval = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[1, 0]]> : tensor<1x2xi64>, undef, l1, sharded>

// -----

// Test: negative index that normalizes to out of bounds (start)
// expected-error @+1 {{collapsed_intervals start index -10 (normalized: -8) is out of bounds for logical_shape rank 2}}
#layout_neg_start_oob = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[-10, 2]]> : tensor<1x2xi64>, undef, l1, sharded>

// -----

// Test: Valid layout (should parse without error)
#layout_valid = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

func.func @test_valid_layout(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_valid>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_valid> {
  return %arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_valid>
}

// RUN: ttmlir-opt %s --split-input-file --verify-diagnostics

// Offset and size are (y, x). Negative offset y.
// expected-error @+1 {{offset values must be non-negative}}
#bad_offset_y = #ttcore.core_range<(-1,0), 1x1>

// -----

// Invalid size rank.
// expected-error @+1 {{size must contain exactly 2 values, got 1}}
#bad_size_rank = #ttcore.core_range<(0,0), 1>

// -----

// Negative offset x.
// expected-error @+1 {{offset values must be non-negative}}
#bad_offset_x = #ttcore.core_range<(0,-1), 1x1>

// -----

// Zero size y.
// expected-error @+1 {{size values must be positive}}
#bad_zero_size_y = #ttcore.core_range<(0,0), 0x1>

// -----

// Zero size x.
// expected-error @+1 {{size values must be positive}}
#bad_zero_size_x = #ttcore.core_range<(0,0), 1x0>

// -----

// DimensionList rejects negative size values before attribute verification.
// expected-error @+1 {{invalid dimension}}
#bad_negative_size_x = #ttcore.core_range<(0,0), 1x-1>

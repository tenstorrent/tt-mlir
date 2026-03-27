// RUN: ttmlir-opt %s --split-input-file --verify-diagnostics

// CoreCoordAttr is (y, x). Negative start y.
// expected-error @+1 {{must be non-negative}}
#bad_start_y = #ttcore.core_range<(-1, 0), (0, 0)>

// -----

// Negative start x.
// expected-error @+1 {{must be non-negative}}
#bad_start_x = #ttcore.core_range<(0, -1), (0, 0)>

// -----

// Negative end y.
// expected-error @+1 {{must be non-negative}}
#bad_end_y = #ttcore.core_range<(0, 0), (-1, 0)>

// -----

// Negative end x (non-negative check runs before start <= end).
// expected-error @+1 {{must be non-negative}}
#bad_end_x = #ttcore.core_range<(0, 0), (0, -1)>

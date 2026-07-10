// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify the ttir.argmax -> D2M decomposition. The lowering builds the index of
// the maximum as:
//   max     = tile_reduce_max(input)
//   eq      = (input == bcast(max))            // 1.0 at matches, 0.0 elsewhere
//   index   = arange (DESCENDING: start=N, step=-1)
//   masked  = eq * index                       // (N - i) at matches, 0 elsewhere
//   reduced = tile_reduce_max(masked)          // = N - smallestMatchingIndex
//   result  = N - reduced                      // recover smallest index (ties -> lowest)
//   typecast to si32
// The descending arange + trailing tile_sub make ties resolve to the SMALLEST
// index, matching torch's argmax.

// Row-wise reduction (reduce the last dim) lowers to ReduceDim::R with a
// row-major descending arange and no transpose.
// CHECK-LABEL: func @argmax_rowwise
func.func @argmax_rowwise(%arg0: tensor<32x32xbf16>) -> tensor<32xsi32> {
  // CHECK: d2m.tile_reduce_max
  // CHECK: d2m.tile_bcast
  // CHECK: d2m.tile_eq
  // CHECK: d2m.arange_block{{.*}}colMajor = false{{.*}}num_elements = 32 : i64, start = 32 : i64, step = -1
  // CHECK: d2m.tile_mul
  // CHECK: d2m.tile_reduce_max
  // The N - result tie-break op.
  // CHECK: d2m.tile_sub
  // CHECK: d2m.tile_typecast
  %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x32xbf16>) -> tensor<32xsi32>
  return %0 : tensor<32xsi32>
}

// Column-wise reduction (reduce the second-to-last dim) lowers to ReduceDim::C
// with a column-major descending arange.
// CHECK-LABEL: func @argmax_colwise
func.func @argmax_colwise(%arg0: tensor<32x32xbf16>) -> tensor<32xsi32> {
  // CHECK: d2m.tile_reduce_max
  // CHECK: d2m.tile_eq
  // CHECK: d2m.arange_block{{.*}}colMajor = true{{.*}}num_elements = 32 : i64, start = 32 : i64, step = -1
  // CHECK: d2m.tile_mul
  // CHECK: d2m.tile_reduce_max
  // CHECK: d2m.tile_sub
  // CHECK: d2m.tile_typecast
  %0 = "ttir.argmax"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<32x32xbf16>) -> tensor<32xsi32>
  return %0 : tensor<32xsi32>
}

// keep_dim=true keeps the reduced dimension as size 1 in the result.
// CHECK-LABEL: func @argmax_keepdim
func.func @argmax_keepdim(%arg0: tensor<32x32xbf16>) -> tensor<32x1xsi32> {
  // CHECK: d2m.tile_reduce_max
  // CHECK: d2m.arange_block
  // CHECK: d2m.tile_sub
  // CHECK: d2m.tile_typecast
  %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x32xbf16>) -> tensor<32x1xsi32>
  return %0 : tensor<32x1xsi32>
}

// An unaligned reduced dimension (40 is not a multiple of the 32 tile width)
// requires masking the out-of-bounds tile lanes with -inf so padding never wins
// the max.
// CHECK-LABEL: func @argmax_unaligned
func.func @argmax_unaligned(%arg0: tensor<32x40xbf16>) -> tensor<32xsi32> {
  // CHECK: d2m.mask {{.*}}logical_shape = [32, 40] fill_value = <neginf>
  // CHECK: d2m.tile_reduce_max
  // CHECK: d2m.arange_block
  // CHECK: d2m.tile_sub
  %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x40xbf16>) -> tensor<32xsi32>
  return %0 : tensor<32xsi32>
}

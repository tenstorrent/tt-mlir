// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// ttir.update_cache lowers directly to ttnn.paged_update_cache. The input is
// permuted from [1, num_heads, num_users, head_dim] to
// [1, num_users, num_heads, head_dim], and a scalar update_index is repeated to
// [num_users].
// CHECK-LABEL: func.func @update_cache_permute_and_repeat
func.func @update_cache_permute_and_repeat(%cache: tensor<4x8x64x128xbf16>, %input: tensor<1x8x4x128xbf16>, %update_index: tensor<1xi32>) -> tensor<4x8x64x128xbf16> {
  // CHECK: "ttnn.permute"{{.*}}permutation = array<i64: 0, 2, 1, 3>
  // CHECK: "ttnn.repeat"{{.*}}repeat_dims = #ttnn.shape<4>
  // CHECK: "ttnn.paged_update_cache"
  "ttir.update_cache"(%cache, %input, %update_index) <{batch_offset = 0 : i32}> : (tensor<4x8x64x128xbf16>, tensor<1x8x4x128xbf16>, tensor<1xi32>) -> ()
  return %cache : tensor<4x8x64x128xbf16>
}

// -----

// update_index is already sized [num_users], so only the input permute is
// inserted (no repeat).
// CHECK-LABEL: func.func @update_cache_permute_only
func.func @update_cache_permute_only(%cache: tensor<4x8x64x128xbf16>, %input: tensor<1x8x4x128xbf16>, %update_index: tensor<4xi32>) -> tensor<4x8x64x128xbf16> {
  // CHECK: "ttnn.permute"{{.*}}permutation = array<i64: 0, 2, 1, 3>
  // CHECK-NOT: "ttnn.repeat"
  // CHECK: "ttnn.paged_update_cache"
  "ttir.update_cache"(%cache, %input, %update_index) <{batch_offset = 0 : i32}> : (tensor<4x8x64x128xbf16>, tensor<1x8x4x128xbf16>, tensor<4xi32>) -> ()
  return %cache : tensor<4x8x64x128xbf16>
}

// -----

// A scalar (rank-0) update_index is reshaped to rank 1 before being repeated to
// [num_users].
// CHECK-LABEL: func.func @update_cache_scalar_index
func.func @update_cache_scalar_index(%cache: tensor<4x8x64x128xbf16>, %input: tensor<1x8x4x128xbf16>, %update_index: tensor<i32>) -> tensor<4x8x64x128xbf16> {
  // CHECK: "ttnn.reshape"{{.*}}shape = [1 : i32]
  // CHECK: "ttnn.repeat"{{.*}}repeat_dims = #ttnn.shape<4>
  // CHECK: "ttnn.paged_update_cache"
  "ttir.update_cache"(%cache, %input, %update_index) <{batch_offset = 0 : i32}> : (tensor<4x8x64x128xbf16>, tensor<1x8x4x128xbf16>, tensor<i32>) -> ()
  return %cache : tensor<4x8x64x128xbf16>
}

// RUN: not ttmlir-opt --ttir-to-ttnn-backend-pipeline %s 2>&1 | FileCheck %s

// ttnn.paged_update_cache has no batch_offset, so a non-zero batch_offset on
// ttir.update_cache must fail to legalize rather than be silently dropped.
// CHECK: error: failed to legalize operation 'ttir.update_cache'
func.func @update_cache_nonzero_batch_offset(%cache: tensor<4x8x64x128xbf16>, %input: tensor<1x8x4x128xbf16>, %update_index: tensor<4xi32>) -> tensor<4x8x64x128xbf16> {
  %0 = "ttir.update_cache"(%cache, %input, %update_index) <{batch_offset = 2 : i32}> : (tensor<4x8x64x128xbf16>, tensor<1x8x4x128xbf16>, tensor<4xi32>) -> tensor<4x8x64x128xbf16>
  return %0 : tensor<4x8x64x128xbf16>
}

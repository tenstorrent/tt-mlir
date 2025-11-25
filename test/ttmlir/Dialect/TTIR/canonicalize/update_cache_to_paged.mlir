// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @update_cache(%arg0: tensor<1x8x16x128xbf16>, %arg1: tensor<1x8x1x128xbf16>, %arg2: tensor<1xui32>) -> tensor<1x8x16x128xbf16> {
  // CHECK: "ttir.paged_update_cache"
  %0 = "ttir.update_cache"(%arg0, %arg1, %arg2) <{batch_offset = 0 : i32}> : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xui32>) -> tensor<1x8x16x128xbf16>
  return %0 : tensor<1x8x16x128xbf16>
}

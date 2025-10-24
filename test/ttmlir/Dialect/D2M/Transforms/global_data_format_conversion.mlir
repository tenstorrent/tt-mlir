// RUN: ttmlir-opt --d2m-global-data-format-conversion="target-format=bf16" %s | FileCheck %s --check-prefix=BF16
// RUN: ttmlir-opt --d2m-global-data-format-conversion="target-format=f32" %s | FileCheck %s --check-prefix=F32
// RUN: ttmlir-opt --d2m-global-data-format-conversion="target-format=bfp_f8" %s | FileCheck %s --check-prefix=BFP8

// BF16-LABEL: func.func @test_add_f32_to_bf16
// F32-LABEL: func.func @test_add_f32_to_bf16
// BFP8-LABEL: func.func @test_add_f32_to_bf16
func.func @test_add_f32_to_bf16(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // BF16: %{{.*}} = ttir.empty() : tensor<64x128xbf16>
  // BF16: %{{.*}} = "ttir.typecast"(%arg1, %{{.*}}) <{conservative_folding = false}> : (tensor<64x128xf32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  // BF16: %{{.*}} = ttir.empty() : tensor<64x128xbf16>
  // BF16: %{{.*}} = "ttir.typecast"(%arg0, %{{.*}}) <{conservative_folding = false}> : (tensor<64x128xf32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  // BF16: %{{.*}} = ttir.empty() : tensor<64x128xbf16>
  // BF16: %{{.*}} = "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}}) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  // BF16: %{{.*}} = ttir.empty() : tensor<64x128xf32>
  // BF16: %{{.*}} = "ttir.typecast"(%{{.*}}, %{{.*}}) <{conservative_folding = false}> : (tensor<64x128xbf16>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // BF16: return %{{.*}} : tensor<64x128xf32>

  // F32: %{{.*}} = ttir.empty() : tensor<64x128xf32>
  // F32: %{{.*}} = "ttir.add"(%arg0, %arg1, %{{.*}}) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // F32: return %{{.*}} : tensor<64x128xf32>

  // BFP8: %{{.*}} = ttir.empty() : tensor<64x128x!ttcore.tile<32x32, bfp_f8>>
  // BFP8: %{{.*}} = "ttir.typecast"(%arg1, %{{.*}}) <{conservative_folding = false}> : (tensor<64x128xf32>, tensor<64x128x!ttcore.tile<32x32, bfp_f8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_f8>>
  // BFP8: %{{.*}} = ttir.empty() : tensor<64x128x!ttcore.tile<32x32, bfp_f8>>
  // BFP8: %{{.*}} = "ttir.typecast"(%arg0, %{{.*}}) <{conservative_folding = false}> : (tensor<64x128xf32>, tensor<64x128x!ttcore.tile<32x32, bfp_f8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_f8>>
  // BFP8: %{{.*}} = ttir.empty() : tensor<64x128x!ttcore.tile<32x32, bfp_f8>>
  // BFP8: %{{.*}} = "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}}) : (tensor<64x128x!ttcore.tile<32x32, bfp_f8>>, tensor<64x128x!ttcore.tile<32x32, bfp_f8>>, tensor<64x128x!ttcore.tile<32x32, bfp_f8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_f8>>
  // BFP8: %{{.*}} = ttir.empty() : tensor<64x128xf32>
  // BFP8: %{{.*}} = "ttir.typecast"(%{{.*}}, %{{.*}}) <{conservative_folding = false}> : (tensor<64x128x!ttcore.tile<32x32, bfp_f8>>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // BFP8: return %{{.*}} : tensor<64x128xf32>

  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

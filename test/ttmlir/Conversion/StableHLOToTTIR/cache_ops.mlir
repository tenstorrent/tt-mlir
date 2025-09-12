// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @update_cache(%arg0: tensor<1x12x16x64xbf16>, %arg1: tensor<1x12x1x64xbf16>, %arg2: tensor<1xi32>) -> tensor<1x12x16x64xbf16> {
    // CHECK: "ttir.update_cache"
    // CHECK: batch_offset = 0 : i32
    %0 = stablehlo.custom_call @tt.update_cache(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {batch_offset = "0"}}: (tensor<1x12x16x64xbf16>, tensor<1x12x1x64xbf16>, tensor<1xi32>) -> tensor<1x12x16x64xbf16>
    return %0 : tensor<1x12x16x64xbf16>
  }

  func.func @fill_cache(%arg0: tensor<1x12x16x64xbf16>, %arg1: tensor<1x12x5x64xbf16>) -> tensor<1x12x16x64xbf16> {
    // CHECK: "ttir.fill_cache"
    // CHECK: batch_offset = 0 : i32
    %0 = stablehlo.custom_call @tt.fill_cache(%arg0, %arg1) {api_version = 0 : i32, mhlo.frontend_attributes = {batch_offset = "0"}} : (tensor<1x12x16x64xbf16>, tensor<1x12x5x64xbf16>) -> tensor<1x12x16x64xbf16>
    return %0 : tensor<1x12x16x64xbf16>
  }
}

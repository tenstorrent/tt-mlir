// RUN: ttmlir-opt %s | FileCheck %s
module {
  func.func @concatenate_heads_llama(%arg0: tensor<1x24x32x128xbf16>) -> tensor<1x32x3072xbf16> {
    %0 = ttir.empty() : tensor<1x32x3072xbf16>
    // CHECK: %{{[0-9]+}} = "ttir.concatenate_heads"
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x24x32x128xbf16>, tensor<1x32x3072xbf16>) -> tensor<1x32x3072xbf16>
    return %1 : tensor<1x32x3072xbf16>
  }
  func.func @concatenate_heads_vit(%arg0: tensor<1x12x197x64xbf16>) -> tensor<1x197x768xbf16> {
    %0 = ttir.empty() : tensor<1x197x768xbf16>
    // CHECK: %{{[0-9]+}} = "ttir.concatenate_heads"
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x12x197x64xbf16>, tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    return %1 : tensor<1x197x768xbf16>
  }
  func.func @concatenate_heads_bert(%arg0: tensor<1x12x256x64xbf16>) -> tensor<1x256x768xbf16> {
    %0 = ttir.empty() : tensor<1x256x768xbf16>
    // CHECK: %{{[0-9]+}} = "ttir.concatenate_heads"
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x12x256x64xbf16>, tensor<1x256x768xbf16>) -> tensor<1x256x768xbf16>
    return %1 : tensor<1x256x768xbf16>
  }
}

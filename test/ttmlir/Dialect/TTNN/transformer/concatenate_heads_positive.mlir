// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @concatenate_heads_llama_1(%arg0: tensor<1x24x32x128xbf16>) -> tensor<1x32x3072xbf16> {
    %0 = ttir.empty() : tensor<1x32x3072xbf16>
    // CHECK: "ttnn.concatenate_heads"(%arg0)
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x24x32x128xbf16>, tensor<1x32x3072xbf16>) -> tensor<1x32x3072xbf16>
    return %1 : tensor<1x32x3072xbf16>
  }
  func.func @concatenate_heads_llama_2(%arg0: tensor<2x24x32x128xbf16>) -> tensor<2x32x3072xbf16> {
    %0 = ttir.empty() : tensor<2x32x3072xbf16>
    // CHECK: "ttnn.concatenate_heads"(%arg0)
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<2x24x32x128xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
    return %1 : tensor<2x32x3072xbf16>
  }
  func.func @concatenate_heads_vit_1(%arg0: tensor<1x12x197x64xbf16>) -> tensor<1x197x768xbf16> {
    %0 = ttir.empty() : tensor<1x197x768xbf16>
    // CHECK: "ttnn.concatenate_heads"(%arg0)
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x12x197x64xbf16>, tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    return %1 : tensor<1x197x768xbf16>
  }
  func.func @concatenate_heads_vit_2(%arg0: tensor<2x12x197x64xbf16>) -> tensor<2x197x768xbf16> {
    %0 = ttir.empty() : tensor<2x197x768xbf16>
    // CHECK: "ttnn.concatenate_heads"(%arg0)
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<2x12x197x64xbf16>, tensor<2x197x768xbf16>) -> tensor<2x197x768xbf16>
    return %1 : tensor<2x197x768xbf16>
  }
  func.func @concatenate_heads_bert_1(%arg0: tensor<1x12x256x64xbf16>) -> tensor<1x256x768xbf16> {
    %0 = ttir.empty() : tensor<1x256x768xbf16>
    // CHECK: "ttnn.concatenate_heads"(%arg0)
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x12x256x64xbf16>, tensor<1x256x768xbf16>) -> tensor<1x256x768xbf16>
    return %1 : tensor<1x256x768xbf16>
  }
  func.func @concatenate_heads_bert_2(%arg0: tensor<2x12x256x64xbf16>) -> tensor<2x256x768xbf16> {
    %0 = ttir.empty() : tensor<2x256x768xbf16>
    // CHECK: "ttnn.concatenate_heads"(%arg0)
    %1 = "ttir.concatenate_heads"(%arg0, %0) : (tensor<2x12x256x64xbf16>, tensor<2x256x768xbf16>) -> tensor<2x256x768xbf16>
    return %1 : tensor<2x256x768xbf16>
  }
}

// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t
// XFAIL: *
// TODO: #3232 re-enable

module {
  func.func @softmax_simple(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    %0 = ttir.empty() : tensor<512x1024xbf16>
    // CHECK: %{{.*}} = linalg.softmax dimension(0) ins(%arg0 : tensor<512x1024xbf16>) outs(%0 : tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    %1 = "ttir.softmax"(%arg0, %0) <{dimension = 0 : si32}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %1 : tensor<512x1024xbf16>
  }

  func.func @softmax(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    %0 = ttir.empty() : tensor<512x1024xbf16>
    // Check for positive dimension attribute
    // CHECK: %{{.*}} = linalg.softmax dimension(1) ins(%arg0 : tensor<512x1024xbf16>) outs(%0 : tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    %1 = "ttir.softmax"(%arg0, %0) <{dimension = 1 : si32}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    %2 = ttir.empty() : tensor<512x1024xbf16>
    // Check for negative dimension attribute
    // CHECK: %{{.*}} = linalg.softmax dimension(1) ins(%1 : tensor<512x1024xbf16>) outs(%2 : tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    %3 = "ttir.softmax"(%1, %2) <{dimension = -1 : si32}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %3 : tensor<512x1024xbf16>
  }
}

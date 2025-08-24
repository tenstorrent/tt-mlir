// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Unit tests for ttir all_gather op

// -----

module attributes {} {
  func.func @all_gather_invalid_dim(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x128xbf16> {
    %0 = ttir.empty() : tensor<1x1x32x128xbf16>
    %1 = "ttir.all_gather"(%arg0, %0) <{all_gather_dim = 4 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x32x32xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    return %1 : tensor<1x1x32x128xbf16>
  }
}
// CHECK: error: 'ttir.all_gather' op Invalid dimension for all gather op. Gather dimension must be >= to input tensor rank or < -input tensor rank, got gather_dim = 4

// -----

module attributes {} {
  func.func @all_gather_invalid_negative_dim(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x128xbf16> {
    %0 = ttir.empty() : tensor<1x1x32x128xbf16>
    %1 = "ttir.all_gather"(%arg0, %0) <{all_gather_dim = -5 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x32x32xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    return %1 : tensor<1x1x32x128xbf16>
  }
}
// CHECK: error: 'ttir.all_gather' op Invalid dimension for all gather op. Gather dimension must be >= to input tensor rank or < -input tensor rank, got gather_dim = -5

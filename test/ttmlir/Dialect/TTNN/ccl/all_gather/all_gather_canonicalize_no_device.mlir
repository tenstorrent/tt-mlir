// RUN: ttmlir-opt --canonicalize --split-input-file %s | FileCheck %s

module attributes {} {
  // CHECK-LABEL: func.func @all_gather_cluster_axis1
  func.func @all_gather_cluster_axis1(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16> {
    // CHECK: "ttnn.all_gather"
    %0 = "ttnn.all_gather"(%arg0) <{all_gather_dim = 3 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
}

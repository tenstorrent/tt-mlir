// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=GreedyL1Interleaved tensor-l1-usage-cap=0.75" -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>, %arg2: tensor<64x96xbf16>, %arg3: tensor<96x32xbf16>, %arg4: tensor<64x32xbf16>) -> tensor<64x32xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_L1:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <{{.*}}>, memref<{{.*}}, #l1>, <interleaved>>
    %0 = ttir.empty() : tensor<64x96xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %2 = ttir.empty() : tensor<64x96xbf16>
    // CHECK: %{{.*}} = "ttnn.linear"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_L1]]>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<64x96xbf16>, tensor<64x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %4 = ttir.empty() : tensor<64x96xbf16>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_L1]]>
    %5 = "ttir.relu"(%3, %4) : (tensor<64x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %6 = ttir.empty() : tensor<64x32xbf16>
    %7 = "ttir.matmul"(%5, %arg3, %6) : (tensor<64x96xbf16>, tensor<96x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    %8 = ttir.empty() : tensor<64x32xbf16>
    // CHECK: %{{.*}} = "ttnn.linear"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_L1]]>
    %9 = "ttir.add"(%7, %arg4, %8) : (tensor<64x32xbf16>, tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    %10 = ttir.empty() : tensor<64x32xbf16>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_L1]]>
    %11 = "ttir.relu"(%9, %10) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %11 : tensor<64x32xbf16>
  }
}

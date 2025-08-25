// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=GreedyL1Interleaved tensor-l1-usage-cap=0.75" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
//       A     B
//        \   /
//          C
//          |
//          D
//
//  (A + C > L1) AND (B + C > L1) AND (A + B > L1) AND (B < A) AND (C < A) AND (A <= L1)
//      =>
//  DRAM: BC; L1: A
//
module attributes {} {
  func.func @forward(%arg0: tensor<5120x5120xbf16>, %arg1: tensor<5120x5120xbf16>, %arg2: tensor<5120x4096xbf16>, %arg3: tensor<5120x4096xbf16>) -> tensor<5120x4096xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK-DAG: #[[LAYOUT_3:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<160x128x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_5:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <{{.*}}>, memref<1x400x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
    %0 = ttir.empty() : tensor<5120x5120xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<5120x5120xbf16, #[[LAYOUT_5]]>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<5120x5120xbf16>, tensor<5120x5120xbf16>, tensor<5120x5120xbf16>) -> tensor<5120x5120xbf16>
    %2 = ttir.empty() : tensor<5120x4096xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<5120x4096xbf16, #[[LAYOUT_3]]>
    %3 = "ttir.add"(%arg2, %arg3, %2) : (tensor<5120x4096xbf16>, tensor<5120x4096xbf16>, tensor<5120x4096xbf16>) -> tensor<5120x4096xbf16>
    %4 = ttir.empty() : tensor<5120x4096xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<5120x4096xbf16, #[[LAYOUT_3]]>
    %5 = "ttir.matmul"(%1, %3, %4) : (tensor<5120x5120xbf16>, tensor<5120x4096xbf16>, tensor<5120x4096xbf16>) -> tensor<5120x4096xbf16>
    return %5 : tensor<5120x4096xbf16>
  }
}

// Test for L1InterleavedFallbackAnalysis: ops should NOT be upgraded to L1 due to L1 space constraints, neither result fits in L1 memory
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1InterleavedTestLargeTensorOutput attributes {} {
  func.func @forward(%arg0: tensor<8192x8192xbf16>, %arg1: tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16> {
    // CHECK-DAG: #[[DRAM_LAYOUT:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<256x256x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

    %0 = ttir.empty() : tensor<8192x8192xbf16>
    // CHECK: "ttnn.relu"{{.*}} -> tensor<8192x8192xbf16, #[[DRAM_LAYOUT]]>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<8192x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>

    // CHECK: "ttnn.add"{{.*}} -> tensor<8192x8192xbf16, #[[DRAM_LAYOUT]]>
    %2 = ttir.empty() : tensor<8192x8192xbf16>
    %3 = "ttir.add"(%1, %arg1, %2) : (tensor<8192x8192xbf16>, tensor<8192x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>

    // As output is the return value, not beneficial to move to L1, will always stay in DRAM.
    // CHECK: "ttnn.add"{{.*}} -> tensor<8192x8192xbf16, #[[DRAM_LAYOUT]]>
    %4 = ttir.empty() : tensor<8192x8192xbf16>
    %5 = "ttir.add"(%3, %arg1, %4) : (tensor<8192x8192xbf16>, tensor<8192x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>

    return %5 : tensor<8192x8192xbf16>
  }
}

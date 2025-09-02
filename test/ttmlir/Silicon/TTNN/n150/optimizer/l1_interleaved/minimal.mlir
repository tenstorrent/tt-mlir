// Test for L1InterleavedFallbackAnalysis: simple no fork no join pattern
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1InterleavedTestMinimal attributes {} {
  func.func @forward(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    // CHECK-DAG: #[[L1_LAYOUT:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>

    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: %{{.*}} = "ttnn.multiply"{{.*}} -> tensor<32x32xbf16, #[[L1_LAYOUT]]>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    %2 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x32xbf16, #[[L1_LAYOUT]]>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    %4 = ttir.empty() : tensor<32x32xbf16>
    // As output is the return value, not beneficial to move to L1, will always stay in DRAM.
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<32x32xbf16, #[[DRAM_LAYOUT]]>
    %5 = "ttir.relu"(%3, %4) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %5 : tensor<32x32xbf16>
  }
}

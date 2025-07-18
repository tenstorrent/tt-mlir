// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true max-legal-layouts=32" -o output_file.mlir %s
// RUN: FileCheck %s --input-file=output_file.mlir
// UNSUPPORTED: Blackhole
// Test isn't supported on blackhole due to file check failure on sharded memory layout:
// https://github.com/tenstorrent/tt-mlir/issues/4187

module attributes {} {
  func.func @forward(%arg0: tensor<64x96xbf16>, %arg1: tensor<96x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_L1:.*]] = #ttnn.ttnn_layout<{{.*}}#[[L1_]]>
    %0 = ttir.empty() : tensor<64x96xbf16>
    // CHECK: {{.*}} = "ttnn.relu"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_L1]]>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %2 = ttir.empty() : tensor<64x64xbf16>
    %3 = "ttir.matmul"(%1, %arg1, %2) : (tensor<64x96xbf16>, tensor<96x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}

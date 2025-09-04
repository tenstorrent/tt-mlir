// Test for L1InterleavedFallbackAnalysis: Check that the row-major workaround doesn't affect the input being in L1 - no crash in runtime.
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1AddMaxPool2D attributes {} {
  func.func @forward(%arg0: tensor<1x64x32x32xbf16>, %arg1: tensor<1x64x32x32xbf16>) -> tensor<1x32x16x32xbf16> {
    // CHECK-DAG: #[[L1_LAYOUT1:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#l1>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT1:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT2:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT3:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT4:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT5:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>

    %0 = ttir.empty() : tensor<1x64x32x32xbf16>
    // Add operation - should be eligible for L1 upgrade
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x64x32x32xbf16, #[[L1_LAYOUT1]]>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x64x32x32xbf16>, tensor<1x64x32x32xbf16>, tensor<1x64x32x32xbf16>) -> tensor<1x64x32x32xbf16>

    %2 = ttir.empty() : tensor<1x32x16x32xbf16>
    // MaxPool2D operation - ignored for L1 upgrade because of row-major workaround
    // CHECK: %{{.*}} = "ttnn.max_pool2d"{{.*}} -> tensor<1x1x512x32xbf16, #[[DRAM_LAYOUT5]]>
    %3 = "ttir.max_pool2d"(%1, %2) <{kernel = array<i32: 2, 2>, stride = array<i32: 2, 2>, padding = array<i32: 0, 0>, dilation = array<i32: 1, 1>, ceil_mode = false}> : (tensor<1x64x32x32xbf16>, tensor<1x32x16x32xbf16>) -> tensor<1x32x16x32xbf16>

    %4 = ttir.empty() : tensor<1x32x16x32xbf16>
    // Not the real model output here, as an additional reshape is the consumer of this op, so might be in L1.
    %5 = "ttir.relu"(%3, %4) : (tensor<1x32x16x32xbf16>, tensor<1x32x16x32xbf16>) -> tensor<1x32x16x32xbf16>

    return %5 : tensor<1x32x16x32xbf16>
  }
}

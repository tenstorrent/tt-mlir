// Test for L1InterleavedFallbackAnalysis: simple fork-join pattern
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true max-legal-layouts=32" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1InterleavedTestForkJoin attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %arg2: tensor<64x128xbf16>, %arg3: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK-DAG: #[[DRAM:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[L1:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#l1>{{.*}}<interleaved>>
    // not immediately consumed -> dram
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[DRAM]]>
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[L1]]>
    %1 = "ttir.add"(%arg2, %arg3) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[L1]]>
    %2 = "ttir.add"(%0, %1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // fork, not a single user -> dram
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<{{.*}}, #[[DRAM]]>
    %3 = "ttir.relu"(%2) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // Fork: %3 used in two ops
    // not immediately consumed -> dram
    // CHECK: %{{.*}} = "ttnn.neg"{{.*}} -> tensor<{{.*}}, #[[DRAM]]>
    %4 = "ttir.neg"(%3) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: %{{.*}} = "ttnn.abs"{{.*}} -> tensor<{{.*}}, #[[L1]]>
    %5 = "ttir.abs"(%3) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // Join: add the results
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[L1]]>
    %6 = "ttir.add"(%4, %5) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // As output is the return value, not beneficial to move to L1, will always stay in DRAM.
    // CHECK: %{{.*}} = "ttnn.abs"{{.*}} -> tensor<{{.*}}, #[[DRAM]]>
    %7 = "ttir.abs"(%6) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %7 : tensor<64x128xbf16>
  }
}

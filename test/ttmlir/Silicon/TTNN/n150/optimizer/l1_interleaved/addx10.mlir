// Test for L1InterleavedFallbackAnalysis: Serves as sanity check for checking perf gain of the analysis for ops valid for it, in this example the Binary Add operation.
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1InterleavedAddx10 attributes {} {
  func.func @forward(%arg0: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"} loc("SimpleAddModel":0:0),
                    %arg1: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "param1"} loc("SimpleAddModel":0:0),
                    %arg2: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "param2"} loc("SimpleAddModel":0:0),
                    %arg3: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "param3"} loc("SimpleAddModel":0:0),
                    %arg4: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "param4"} loc("SimpleAddModel":0:0),
                    %arg5: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "param5"} loc("SimpleAddModel":0:0),
                    %arg6: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "param6"} loc("SimpleAddModel":0:0),
                    %arg7: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "param7"} loc("SimpleAddModel":0:0),
                    %arg8: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "param8"} loc("SimpleAddModel":0:0),
                    %arg9: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "param9"} loc("SimpleAddModel":0:0),
                    %arg10: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "param10"} loc("SimpleAddModel":0:0)) -> (tensor<32x4096xbf16> {ttir.name = "SimpleAddModel.output"}) {

    // CHECK-DAG: #[[L1_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#l1>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>

    // Add operations should be upgraded to L1 interleaved when beneficial
    %0 = ttir.empty() : tensor<32x4096xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x4096xbf16, #[[L1_LAYOUT]]>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>

    %2 = ttir.empty() : tensor<32x4096xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x4096xbf16, #[[L1_LAYOUT]]>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>

    %4 = ttir.empty() : tensor<32x4096xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x4096xbf16, #[[L1_LAYOUT]]>
    %5 = "ttir.add"(%3, %arg3, %4) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %6 = ttir.empty() : tensor<32x4096xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x4096xbf16, #[[L1_LAYOUT]]>
    %7 = "ttir.add"(%5, %arg4, %6) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>

    %8 = ttir.empty() : tensor<32x4096xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x4096xbf16, #[[L1_LAYOUT]]>
    %9 = "ttir.add"(%7, %arg5, %8) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %10 = ttir.empty() : tensor<32x4096xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x4096xbf16, #[[L1_LAYOUT]]>
    %11 = "ttir.add"(%9, %arg6, %10) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>

    %12 = ttir.empty() : tensor<32x4096xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x4096xbf16, #[[L1_LAYOUT]]>
    %13 = "ttir.add"(%11, %arg7, %12) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %14 = ttir.empty() : tensor<32x4096xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x4096xbf16, #[[L1_LAYOUT]]>
    %15 = "ttir.add"(%13, %arg8, %14) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>

    %16 = ttir.empty() : tensor<32x4096xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x4096xbf16, #[[L1_LAYOUT]]>
    %17 = "ttir.add"(%15, %arg9, %16) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>

    %18 = ttir.empty() : tensor<32x4096xbf16>
    // As output is the return value, not beneficial to move to L1, will always stay in DRAM.
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<32x4096xbf16, #[[DRAM_LAYOUT]]>
    %19 = "ttir.add"(%17, %arg10, %18) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %19 : tensor<32x4096xbf16>
  }
}

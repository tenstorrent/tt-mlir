// Test for L1InterleavedFallbackAnalysis: Ops Typecast and Slice are currently not properly supported for L1 interleaved input nor output, as in that layout they break during runtime.
// Comparison operations are fine for L1 interleaved if consumed by a different op, showcased by the Abs operation.
// Example ops inspired by the YOLO model, can be used to check these ops when fixed as a trial for YOLO.
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1InterleavedRuntimeSupportTypecastSlice attributes {} {
  func.func @forward(%arg0: tensor<1x144x8400xbf16>, %arg1: tensor<1x64x8400xbf16>) -> tensor<1x16x8400xf32> {
    // CHECK-DAG: #[[DRAM_LAYOUT1:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT2:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT3:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT4:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT5:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[L1_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#l1>{{.*}}<interleaved>>


    // slice_static operation - not supported runtime upgrade to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x64x8400xbf16, #[[DRAM_LAYOUT2]]>
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 64 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x144x8400xbf16>) -> tensor<1x64x8400xbf16>

    // Greater than comparison - because of consumer being Typecast (not supported for input upgrade), not upgraded to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.gt"{{.*}} -> tensor<1x64x8400xbf16, #[[DRAM_LAYOUT2]]>
    %1 = "ttir.gt"(%0, %arg1) : (tensor<1x64x8400xbf16>, tensor<1x64x8400xbf16>) -> tensor<1x64x8400xbf16>

    // Typecast operation - not supported runtime upgrade to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.typecast"{{.*}} -> tensor<1x64x8400xf32, #[[DRAM_LAYOUT4]]>
    %2 = "ttir.typecast"(%1) : (tensor<1x64x8400xbf16>) -> tensor<1x64x8400xf32>

    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %3 = "ttir.slice_static"(%2) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x8400xf32>) -> tensor<1x32x8400xf32>

    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x32x8400xbf16, #[[DRAM_LAYOUT5]]>
    %4 = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x8400xbf16>) -> tensor<1x32x8400xbf16>

    // CHECK: %{{.*}} = "ttnn.typecast"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %5 = "ttir.typecast"(%4) : (tensor<1x32x8400xbf16>) -> tensor<1x32x8400xf32>

    // Less than comparison - not upgraded to L1 because not immediately consumed by its user (schedule)
    // CHECK: %{{.*}} = "ttnn.lt"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %6 = "ttir.lt"(%3, %5) : (tensor<1x32x8400xf32>, tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x32x8400xbf16, #[[DRAM_LAYOUT5]]>
    %7 = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 32 : i32, 0 : i32], ends = [1 : i32, 64 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x8400xbf16>) -> tensor<1x32x8400xbf16>

    // CHECK: %{{.*}} = "ttnn.typecast"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %8 = "ttir.typecast"(%7) : (tensor<1x32x8400xbf16>) -> tensor<1x32x8400xf32>

    // Greater than comparison - because of consumer being Abs (supported for input upgrade), upgraded to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.gt"{{.*}} -> tensor<1x32x8400xf32, #[[L1_LAYOUT]]>
    %9 = "ttir.gt"(%6, %8) : (tensor<1x32x8400xf32>, tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    // Abs operation - because of consumer being Typecast (not supported for input upgrade), not upgraded to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.abs"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %10 = "ttir.abs"(%9) : (tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x32x8400xbf16, #[[DRAM_LAYOUT5]]>
    %11 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 96 : i32, 0 : i32], ends = [1 : i32, 128 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x144x8400xbf16>) -> tensor<1x32x8400xbf16>

    // CHECK: %{{.*}} = "ttnn.typecast"{{.*}} -> tensor<1x32x8400xbf16, #[[DRAM_LAYOUT5]]>
    %12 = "ttir.typecast"(%10) : (tensor<1x32x8400xf32>) -> tensor<1x32x8400xbf16>

    // Greater than or equal comparison - because of consumer being Typecast (not supported for input upgrade), not upgraded to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.ge"{{.*}} -> tensor<1x32x8400xbf16, #[[DRAM_LAYOUT5]]>
    %13 = "ttir.ge"(%12, %11) : (tensor<1x32x8400xbf16>, tensor<1x32x8400xbf16>) -> tensor<1x32x8400xbf16>

    // CHECK: %{{.*}} = "ttnn.typecast"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %14 = "ttir.typecast"(%13) : (tensor<1x32x8400xbf16>) -> tensor<1x32x8400xf32>

    // Add operation - because of consumer being slice_static (not supported for input upgrade), not upgraded to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %15 = "ttir.add"(%14, %3) : (tensor<1x32x8400xf32>, tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    // As output is the return value, not beneficial to move to L1, will always stay in DRAM anyway.
    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x16x8400xf32, #[[DRAM_LAYOUT3]]>
    %16 = "ttir.slice_static"(%15) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x8400xf32>) -> tensor<1x16x8400xf32>

    return %16 : tensor<1x16x8400xf32>
  }
}

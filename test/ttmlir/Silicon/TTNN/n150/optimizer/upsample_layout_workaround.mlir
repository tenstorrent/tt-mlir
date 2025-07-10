// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

// CHECK-DAG: #[[INPUT_LAYOUT:.*]] = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 400 + d1 * 20 + d2, d3), <1x1>, memref<400x640xbf16, #dram>, <interleaved>>
// CHECK-DAG: #[[OUTPUT_LAYOUT:.*]] = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1600 + d1 * 40 + d2, d3), <1x1>, memref<1600x640xbf16, #dram>, <interleaved>>

func.func @main(%arg0: tensor<1x20x20x640xbf16>) -> tensor<1x40x40x640xbf16> {
    %0 = ttir.empty() : tensor<1x40x40x640xbf16>
    // CHECK: ttnn.upsample
    // CHECK-SAME: tensor<1x20x20x640xbf16, #[[INPUT_LAYOUT]]>
    // CHECK-SAME: -> tensor<1x40x40x640xbf16, #[[OUTPUT_LAYOUT]]>
    %1 = "ttir.upsample2d"(%arg0, %0) <{mode = "nearest", scale_factor = 2 : si32}> {channel_last = true} : (tensor<1x20x20x640xbf16>, tensor<1x40x40x640xbf16>) -> tensor<1x40x40x640xbf16>
    return %1 : tensor<1x40x40x640xbf16>
}

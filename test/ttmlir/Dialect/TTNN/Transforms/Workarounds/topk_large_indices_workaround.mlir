// RUN: ttmlir-opt --ttcore-register-device="mock-system-desc-arch=blackhole" --ttnn-layout --convert-ttir-to-ttnn --ttnn-resolve-composites="composite-resolution=force-promote" --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// The tt-metal topk_large_indices kernel requires a ROW_MAJOR bf16 input and
// produces a ROW_MAJOR ui32 output. The TTNNWorkaround pass inserts
// ttnn.to_layout ops so the promoted ttnn.topk_large_indices runs on
// ROW_MAJOR (untiled) operands.

// ROW_MAJOR layouts are the ones whose memref has plain (non-tile) elements.
// CHECK-DAG: #[[ROW_BF16:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x512xbf16, #dram>{{.*}}>
// CHECK-DAG: #[[ROW_U32:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x512xui32, #dram>{{.*}}>

module {
  func.func @topk_large_indices(%input: tensor<1x512xbf16>) -> tensor<1x512xui32> {
    // CHECK-LABEL: @topk_large_indices
    // CHECK: "ttnn.to_layout"
    // CHECK: %[[IDX:[0-9]+]] = "ttnn.topk_large_indices"
    // CHECK-SAME: <{k = 512 : ui32}>
    // CHECK-SAME: (tensor<1x512xbf16, #[[ROW_BF16]]>) -> tensor<1x512xui32, #[[ROW_U32]]>
    // CHECK: "ttnn.to_layout"(%[[IDX]])
    %0 = "ttcore.composite"(%input) <{composite_name = "topk_large_indices", decomposition = @decomp, composite_attributes = {k = 512 : ui32}}> : (tensor<1x512xbf16>) -> tensor<1x512xui32>
    return %0 : tensor<1x512xui32>
  }
  func.func private @decomp(%input: tensor<1x512xbf16>) -> tensor<1x512xui32> {
    %values, %indices = "ttir.topk"(%input) <{k = 512 : i32, dim = -1 : i32, largest = true, sorted = true}> : (tensor<1x512xbf16>) -> (tensor<1x512xbf16>, tensor<1x512xui32>)
    return %indices : tensor<1x512xui32>
  }
}

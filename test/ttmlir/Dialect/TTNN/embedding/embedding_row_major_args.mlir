// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s -mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// Verify that arguments consumed only by ttnn.embedding arrive row-major, so
// there is no per-step ttnn.to_layout on the embedding table.

// CHECK-LABEL: func.func @embedding_untagged_args
// CHECK-SAME: %arg1: tensor<8192x512xbf16, #ttnn.ttnn_layout<{{.*}}memref<8192x512xbf16,{{.*}}>
// CHECK-NOT: "ttnn.to_layout"
// CHECK: = "ttnn.embedding"(%arg0, %arg1)
func.func @embedding_untagged_args(%indices: tensor<1x128xui32>,
                                   %table: tensor<8192x512xbf16>) -> tensor<1x128x512xbf16> {
    %0 = "ttir.embedding"(%indices, %table) : (tensor<1x128xui32>, tensor<8192x512xbf16>) -> tensor<1x128x512xbf16>
    return %0 : tensor<1x128x512xbf16>
}

// A frozen table is tagged as a parameter, which on its own leaves the argument
// tiled. The consumer check runs first, so it still arrives row-major.

// CHECK-LABEL: func.func @embedding_parameter_table
// CHECK-SAME: %arg1: tensor<8192x512xbf16, #ttnn.ttnn_layout<{{.*}}memref<8192x512xbf16,{{.*}}>
// CHECK-NOT: "ttnn.to_layout"
// CHECK: = "ttnn.embedding"(%arg0, %arg1)
func.func @embedding_parameter_table(%indices: tensor<1x128xui32> {ttcore.argument_type = #ttcore.argument_type<input>},
                                     %table: tensor<8192x512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x128x512xbf16> {
    %0 = "ttir.embedding"(%indices, %table) : (tensor<1x128xui32>, tensor<8192x512xbf16>) -> tensor<1x128x512xbf16>
    return %0 : tensor<1x128x512xbf16>
}

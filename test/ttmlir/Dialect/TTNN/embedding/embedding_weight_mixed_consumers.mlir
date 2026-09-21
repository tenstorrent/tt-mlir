// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s -mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// The row-major boundary layout only fires when all consumers agree. Here the
// table is also a matmul weight, so it stays tiled and the embedding gets its
// row-major copy from a ttnn.to_layout inside the graph.

// CHECK-LABEL: func.func @embedding_and_matmul_share_weight
// CHECK-SAME: %arg1: tensor<512x512xbf16, #ttnn.ttnn_layout<{{.*}}!ttcore.tile<32x32, bf16>,{{.*}}>
// CHECK: %[[RM:.*]] = "ttnn.to_layout"(%arg1)
// CHECK: = "ttnn.embedding"(%arg0, %[[RM]])
// CHECK: = "ttnn.matmul"(%arg2, %arg1)
func.func @embedding_and_matmul_share_weight(%indices: tensor<1x128xui32>,
                                             %table: tensor<512x512xbf16>,
                                             %activations: tensor<128x512xbf16>) -> (tensor<1x128x512xbf16>, tensor<128x512xbf16>) {
    %0 = "ttir.embedding"(%indices, %table) : (tensor<1x128xui32>, tensor<512x512xbf16>) -> tensor<1x128x512xbf16>
    %1 = "ttir.matmul"(%activations, %table) : (tensor<128x512xbf16>, tensor<512x512xbf16>) -> tensor<128x512xbf16>
    return %0, %1 : tensor<1x128x512xbf16>, tensor<128x512xbf16>
}

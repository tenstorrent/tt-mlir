// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true max-legal-layouts=32" -o output_file.mlir %s
// RUN: FileCheck %s --input-file=output_file.mlir
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>, %arg2: tensor<96x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_4:ttnn_layout4]] = #ttnn.ttnn_layout<{{.*}}, memref<{{.*}}, #l1_>, {{.*}}>
    %0 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: {{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_4]]>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %2 = tensor.empty() : tensor<64x64xbf16>
    %3 = "ttir.matmul"(%1, %arg2, %2) : (tensor<64x96xbf16>, tensor<96x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}

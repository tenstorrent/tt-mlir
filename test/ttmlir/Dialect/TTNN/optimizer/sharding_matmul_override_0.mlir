// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true max-legal-layouts=0" %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>, %arg2: tensor<96x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: #[[LAYOUT_7:ttnn_layout7]] = #ttnn.ttnn_layout<{{.*}}, memref<{{.*}}, #dram>, {{.*}}>
    %0 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: {{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_7]]>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %2 = tensor.empty() : tensor<64x64xbf16>
    // CHECK: {{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x64xbf16, #[[LAYOUT_7]]>
    %3 = "ttir.matmul"(%1, %arg2, %2) : (tensor<64x96xbf16>, tensor<96x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}

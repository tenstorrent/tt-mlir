// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true max-legal-layouts=0" -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-NOT: #ttnn.ttnn_layout<{{.*}}, memref<{{.*}}, #l1>, {{.*}}>
// CHECK-DAG: #[[LAYOUT_0:.*]] = #ttnn.ttnn_layout<{{.*}}, memref<2x3x!ttcore.tile<32x32, bf16>, #dram>, {{.*}}>
// CHECK-DAG: #[[LAYOUT_1:.*]] = #ttnn.ttnn_layout<{{.*}}, memref<2x2x!ttcore.tile<32x32, bf16>, #dram>, {{.*}}>
// CHECK-NOT: #ttnn.ttnn_layout<{{.*}}, memref<{{.*}}, #l1>, {{.*}}>

module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>, %arg2: tensor<96x64xbf16>) -> tensor<64x64xbf16> {
    %0 = ttir.empty() : tensor<64x96xbf16>
    // CHECK: {{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_0]]>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %2 = ttir.empty() : tensor<64x64xbf16>
    // CHECK: {{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x64xbf16, #[[LAYOUT_1]]>
    %3 = "ttir.matmul"(%1, %arg2, %2) : (tensor<64x96xbf16>, tensor<96x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}

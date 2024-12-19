// RUN: ttmlir-opt --mlir-print-debuginfo --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true override-output-layout=matmul_1_in_1_layout=1x1:l1:interleaved:tile:bf16" %s | FileCheck %s
#loc = loc("Matmul":4294967295:0)
// CHECK-DAG: #[[LOC_MATMUL_IN0:.*]] = loc("matmul_1_in_0_layout"(#loc3))
// CHECK-DAG: #[[LOC_MATMUL_IN1:.*]] = loc("matmul_1_in_1_layout"(#loc3))
// CHECK-DAG: #[[LOC_MATMUL:.*]] = loc("matmul_1"(#loc3))
// CHECK-DAG: #[[IN_1_LAYOUT:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x12x!tt.tile<32x32, bf16>, #l1_>, <interleaved>>
// XFAIL: *
// TODO: Layout override on the optimizer needs update after default input/output tile layout.
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
    %0 = tensor.empty() : tensor<64x96xbf16> loc(#loc2)
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} loc(#[[LOC_MATMUL]])
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16> loc(#loc2)
    return %1 : tensor<64x96xbf16>
  } loc(#loc)
} loc(#loc)

#loc1 = loc("Matmul":4294967295:1)
#loc2 = loc("matmul_1"(#loc1))

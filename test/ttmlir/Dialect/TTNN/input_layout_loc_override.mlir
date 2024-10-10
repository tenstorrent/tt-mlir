// RUN: ttmlir-opt --mlir-print-debuginfo --ttir-to-ttnn-backend-pipeline="enable-optimizer=true sharding-pass-enabled=true override-output-layout=matmul_1_in_1_layout=1x1:l1:interleaved" %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#loc = loc("Matmul":4294967295:0)
// CHECK-DAG: #[[loc_matmul_in0:.*]] = loc("matmul_1_in_0_layout"(#loc3))
// CHECK-DAG: #[[loc_matmul_in1:.*]] = loc("matmul_1_in_1_layout"(#loc3))
// CHECK-DAG: #[[loc_matmul:.*]] = loc("matmul_1"(#loc3))
// CHECK-DAG: #[[in_1_layout:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<128x96xbf16, #l1_>, interleaved>

module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
    %0 = tensor.empty() : tensor<64x96xbf16> loc(#loc2)
    // CHECK-DAG: %[[C:.*]] = "ttnn.to_device"[[C:.*]] loc(#[[loc_matmul_in0]])
    // CHECK-DAG: %[[C:.*]] = "ttnn.to_device"[[C:.*]] -> tensor<128x96xbf16, #[[in_1_layout]]> loc(#[[loc_matmul_in1]])
    // CHECK-DAG: %[[C:.*]] = "ttnn.matmul"[[C:.*]] loc(#[[loc_matmul]])
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16> loc(#loc2)
    return %1 : tensor<64x96xbf16>
  } loc(#loc)
} loc(#loc)

#loc1 = loc("Matmul":4294967295:1)
#loc2 = loc("matmul_1"(#loc1))

// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=BFInterleaved tensor-l1-usage-cap=0.75" -o %t %s
// RUN: FileCheck %s --input-file=%t
// XFAIL: *
#any_device = #ttcore.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<6144x6144xbf16>, %arg1: tensor<6144x6144xbf16>, %arg2: tensor<6144x6144xbf16>) -> tensor<6144x6144xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    %0 = ttir.empty() : tensor<6144x6144xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<6144x6144xbf16>, tensor<6144x6144xbf16>, tensor<6144x6144xbf16>) -> tensor<6144x6144xbf16>
    %2 = ttir.empty() : tensor<6144x6144xbf16>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<6144x6144xbf16>, tensor<6144x6144xbf16>, tensor<6144x6144xbf16>) -> tensor<6144x6144xbf16>
    return %3 : tensor<6144x6144xbf16>
  }
}

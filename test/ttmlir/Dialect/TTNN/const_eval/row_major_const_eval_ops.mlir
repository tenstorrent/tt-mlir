// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=false enable-row-major-const-eval-ops=true" %s | FileCheck %s

module {
  // CHECK-LABEL: forward_const_eval_0
  // 1. This function will only have two arguments, weight and scale.
  //    Weight is already in row_major so we will only have one to_layout op.
  // CHECK: to_layout
  // CHECK-SAME: layout = #ttnn.layout<row_major>

  // CHECK-LABEL: forward(
  func.func @forward(%arg0: tensor<64x64x3x3xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.conv2d_weight}, %arg1: tensor<1x1x1x64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x64xf32>, %arg3: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
    %0 = ttir.empty() : tensor<1x56x56x64xf32>
    %1 = ttir.empty() : tensor<64x1x1x1xf32>
    %2 = "ttir.reshape"(%arg1, %1) <{shape = [64 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x64xf32>, tensor<64x1x1x1xf32>) -> tensor<64x1x1x1xf32>
    %3 = ttir.empty() : tensor<64x64x3x3xf32>
    %4 = "ttir.multiply"(%arg0, %2, %3) : (tensor<64x64x3x3xf32>, tensor<64x1x1x1xf32>, tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %5 = "ttir.conv2d"(%arg3, %4, %arg2, %0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xf32>, tensor<64x64x3x3xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %6 = ttir.empty() : tensor<1x56x56x64xf32>
    %7 = "ttir.relu"(%5, %6) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    return %7 : tensor<1x56x56x64xf32>
  }
}

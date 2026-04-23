// RUN: ttmlir-opt --ttir-strip-intermediate-ttnn-layouts %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
                                  memref<1x1x!ttcore.tile<32x32, bf16>, #dram>,
                                  <interleaved>>

// CHECK-LABEL: func.func @strip_single_intermediate
func.func @strip_single_intermediate(
    %arg0: tensor<32x32xbf16, #ttnn_layout>,
    %arg1: tensor<32x32xbf16, #ttnn_layout>)
    -> tensor<32x32xbf16, #ttnn_layout> {
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout>, %arg1: tensor<32x32xbf16, #ttnn_layout>)
  // CHECK-SAME: -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %arg1)
  // CHECK-SAME: -> tensor<32x32xbf16>{{$}}
  %0 = "ttir.add"(%arg0, %arg1)
      : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>)
      -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[MUL:.*]] = "ttir.multiply"(%[[ADD]], %arg0)
  // CHECK-SAME: -> tensor<32x32xbf16, #ttnn_layout>
  %1 = "ttir.multiply"(%0, %arg0)
      : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>)
      -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: return %[[MUL]] : tensor<32x32xbf16, #ttnn_layout>
  return %1 : tensor<32x32xbf16, #ttnn_layout>
}

// CHECK-LABEL: func.func @strip_eltwise_unary_chain
func.func @strip_eltwise_unary_chain(%arg0: tensor<32x32xbf16, #ttnn_layout>)
    -> tensor<32x32xbf16, #ttnn_layout> {
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0)
  // CHECK-SAME: -> tensor<32x32xbf16>{{$}}
  %0 = "ttir.abs"(%arg0)
      : (tensor<32x32xbf16, #ttnn_layout>)
      -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[NEG:.*]] = "ttir.neg"(%[[ABS]])
  // CHECK-SAME: -> tensor<32x32xbf16>{{$}}
  %1 = "ttir.neg"(%0)
      : (tensor<32x32xbf16, #ttnn_layout>)
      -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[EXP:.*]] = "ttir.exp"(%[[NEG]])
  // CHECK-SAME: -> tensor<32x32xbf16>{{$}}
  %2 = "ttir.exp"(%1)
      : (tensor<32x32xbf16, #ttnn_layout>)
      -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[SIG:.*]] = "ttir.sigmoid"(%[[EXP]])
  // CHECK-SAME: -> tensor<32x32xbf16>{{$}}
  %3 = "ttir.sigmoid"(%2)
      : (tensor<32x32xbf16, #ttnn_layout>)
      -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[TANH:.*]] = "ttir.tanh"(%[[SIG]])
  // CHECK-SAME: -> tensor<32x32xbf16, #ttnn_layout>
  %4 = "ttir.tanh"(%3)
      : (tensor<32x32xbf16, #ttnn_layout>)
      -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: return %[[TANH]] : tensor<32x32xbf16, #ttnn_layout>
  return %4 : tensor<32x32xbf16, #ttnn_layout>
}

// DPS init operands of toLayout ops must also be preserved, since their type is coupled to the result type by the verifier.
// CHECK-LABEL: func.func @strip_with_to_layout
func.func @strip_with_to_layout(
    %arg0: tensor<32x32xbf16, #ttnn_layout>)
    -> tensor<32x32xbf16, #ttnn_layout> {
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[ABS:.*]] = "ttir.abs"(%arg0)
  // CHECK-SAME: (tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16>{{$}}
  %0 = "ttir.abs"(%arg0)
      : (tensor<32x32xbf16, #ttnn_layout>)
      -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[EMPTY:.*]] = ttir.empty() : tensor<32x32xbf16, #ttnn_layout>
  %1 = ttir.empty() : tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[TO:.*]] = ttir.to_layout %[[ABS]], %[[EMPTY]]
  // CHECK-SAME: : tensor<32x32xbf16> into tensor<32x32xbf16, #ttnn_layout> -> tensor<32x32xbf16, #ttnn_layout>
  %2 = ttir.to_layout %0, %1
      : tensor<32x32xbf16, #ttnn_layout>
        into tensor<32x32xbf16, #ttnn_layout>
        -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: return %[[TO]] : tensor<32x32xbf16, #ttnn_layout>
  return %2 : tensor<32x32xbf16, #ttnn_layout>
}

// Two (empty, to_layout) sequences in series.
// CHECK-LABEL: func.func @strip_with_two_to_layouts
func.func @strip_with_two_to_layouts(
    %arg0: tensor<32x32xbf16, #ttnn_layout>,
    %arg1: tensor<32x32xbf16, #ttnn_layout>)
    -> tensor<32x32xbf16, #ttnn_layout> {
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout>, %arg1: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %arg1)
  // CHECK-SAME: -> tensor<32x32xbf16>{{$}}
  %0 = "ttir.add"(%arg0, %arg1)
      : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>)
      -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[E1:.*]] = ttir.empty() : tensor<32x32xbf16, #ttnn_layout>
  %1 = ttir.empty() : tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[T1:.*]] = ttir.to_layout %[[ADD]], %[[E1]]
  // CHECK-SAME: : tensor<32x32xbf16> into tensor<32x32xbf16, #ttnn_layout> -> tensor<32x32xbf16, #ttnn_layout>
  %2 = ttir.to_layout %0, %1
      : tensor<32x32xbf16, #ttnn_layout>
        into tensor<32x32xbf16, #ttnn_layout>
        -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[MUL:.*]] = "ttir.multiply"(%[[T1]], %arg0)
  // CHECK-SAME: -> tensor<32x32xbf16>{{$}}
  %3 = "ttir.multiply"(%2, %arg0)
      : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>)
      -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[E2:.*]] = ttir.empty() : tensor<32x32xbf16, #ttnn_layout>
  %4 = ttir.empty() : tensor<32x32xbf16, #ttnn_layout>
  // CHECK: %[[T2:.*]] = ttir.to_layout %[[MUL]], %[[E2]]
  // CHECK-SAME: : tensor<32x32xbf16> into tensor<32x32xbf16, #ttnn_layout> -> tensor<32x32xbf16, #ttnn_layout>
  %5 = ttir.to_layout %3, %4
      : tensor<32x32xbf16, #ttnn_layout>
        into tensor<32x32xbf16, #ttnn_layout>
        -> tensor<32x32xbf16, #ttnn_layout>
  // CHECK: return %[[T2]] : tensor<32x32xbf16, #ttnn_layout>
  return %5 : tensor<32x32xbf16, #ttnn_layout>
}

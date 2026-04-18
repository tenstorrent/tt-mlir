// RUN: ttmlir-opt --ttir-strip-intermediate-ttnn-layouts %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
                                  memref<1x1x!ttcore.tile<32x32, bf16>, #dram>,
                                  <interleaved>>

// Intermediate result has its ttnn_layout stripped; args/return-feeder preserved.
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

// Five-op unary chain: every intermediate is stripped; only the return-feeder keeps its encoding.
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

// DPS init operands of preserved DestinationStyleOpInterface ops must also be
// preserved, since their type is coupled to the result type by the verifier.
// In this case, %1 (ttir.empty) is the DPS init of %2 (ttir.to_layout),
// so %1's encoding must survive even though %1 is an intermediate.
// Only %0 (ttir.abs result) should have its encoding stripped.
// CHECK-LABEL: func.func @strip_with_to_layout_dps
func.func @strip_with_to_layout_dps(
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

// RUN: ttmlir-opt --ttcore-register-device --ttcore-wrap-device-module --ttnn-d2m-fusing %s | FileCheck %s

// Layout attributes required for binary ops
#l1 = #ttnn.buffer_type<l1>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x8x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

module {
  // CHECK-LABEL: func.func @long_chain
  func.func @long_chain(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.log"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.neg"(%2) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %4 = "ttnn.abs"(%3) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %5 = "ttnn.sigmoid"(%4) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %5 : tensor<64x256xbf16, #layout>
  }

  // CHECK-LABEL: func.func @simple_chain_two_producers
  func.func @simple_chain_two_producers(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>, %arg2: tensor<64x128xbf16, #layout>, %arg3: tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.matmul"(%arg2, %arg3) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.add"(%0, %1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x256xbf16, #layout>, tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.exp"(%2) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %4 = "ttnn.log"(%3) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %4 : tensor<64x256xbf16, #layout>
  }

  // CHECK-LABEL: func.func @chain_with_middle_exit
  func.func @chain_with_middle_exit(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>, %arg2: tensor<256x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.log"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.neg"(%2) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    // exp also feeds this matmul so it is excluded from chain
    %4 = "ttnn.matmul"(%1, %arg2) : (tensor<64x256xbf16, #layout>, tensor<256x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %3 : tensor<64x256xbf16, #layout>
  }

  // CHECK-LABEL: func.func @chain_with_middle_entry
  func.func @chain_with_middle_entry(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>, %arg2: tensor<64x128xbf16, #layout>, %arg3: tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.matmul"(%arg2, %arg3) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.neg"(%2) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    // add takes neg from eltwise chain and matmul2 as a non-eltwise middle entry
    %4 = "ttnn.add"(%3, %1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x256xbf16, #layout>, tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %5 = "ttnn.log"(%4) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %5 : tensor<64x256xbf16, #layout>
  }

  // CHECK-LABEL: func.func @join_pattern_into_chain
  func.func @join_pattern_into_chain(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>, %arg2: tensor<64x128xbf16, #layout>, %arg3: tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.matmul"(%arg2, %arg3) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.neg"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %4 = "ttnn.add"(%2, %3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x256xbf16, #layout>, tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %5 = "ttnn.log"(%4) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %5 : tensor<64x256xbf16, #layout>
  }

  // CHECK-LABEL: func.func @diamond_pattern
  func.func @diamond_pattern(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.neg"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.abs"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %4 = "ttnn.add"(%2, %3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x256xbf16, #layout>, tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %5 = "ttnn.log"(%4) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %5 : tensor<64x256xbf16, #layout>
  }

  // CHECK-LABEL: func.func @diamond_with_non_eltwise_consumer
  func.func @diamond_with_non_eltwise_consumer(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>, %arg2: tensor<256x256xbf16, #layout>) -> (tensor<64x256xbf16, #layout>, tensor<64x256xbf16, #layout>) {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.neg"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.abs"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %4 = "ttnn.add"(%2, %3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x256xbf16, #layout>, tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %5 = "ttnn.log"(%4) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    // %1 exp also feeds matmul, excluding from chain; %2 %3 are now the entry ops into the chain.
    %6 = "ttnn.matmul"(%1, %arg2) : (tensor<64x256xbf16, #layout>, tensor<256x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %5, %6 : tensor<64x256xbf16, #layout>, tensor<64x256xbf16, #layout>
  }

  // CHECK-LABEL: func.func @single_eltwise_no_chain
  func.func @single_eltwise_no_chain(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.relu"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %1 : tensor<64x256xbf16, #layout>
  }

}

// RUN: ttmlir-opt --ttcore-register-device --ttcore-wrap-device-module --ttnn-d2m-fusing %s | FileCheck %s


#l1 = #ttnn.buffer_type<l1>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x8x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
module {
  // CHECK-LABEL: func.func @long_chain
  func.func @long_chain(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    // CHECK: %[[MATMUL:.*]] = "ttnn.matmul"
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"
    // CHECK: ttnn.d2m_subgraph @d2m_subgraph_0
    // CHECK-NEXT: ins(%[[MATMUL]] : tensor<64x256xbf16
    // CHECK-NEXT: outs(%[[EMPTY]] : tensor<64x256xbf16
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
    // CHECK: %[[MM1:.*]] = "ttnn.matmul"
    // CHECK: %[[MM2:.*]] = "ttnn.matmul"
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"
    // CHECK: ttnn.d2m_subgraph @d2m_subgraph_1
    // CHECK-NEXT: ins(%[[MM1]], %[[MM2]] :
    // CHECK-NEXT: outs(%[[EMPTY]] :
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.matmul"(%arg2, %arg3) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.add"(%0, %1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x256xbf16, #layout>, tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.exp"(%2) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %4 = "ttnn.log"(%3) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %4 : tensor<64x256xbf16, #layout>
  }

  // CHECK-LABEL: func.func @chain_with_middle_exit
  func.func @chain_with_middle_exit(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>, %arg2: tensor<256x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    // CHECK: "ttnn.matmul"
    // CHECK: %[[EXP:.*]] = "ttnn.exp"
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"
    // CHECK: ttnn.d2m_subgraph @d2m_subgraph_2
    // CHECK-NEXT: ins(%[[EXP]] :
    // CHECK-NEXT: outs(%[[EMPTY]] :
    // CHECK: "ttnn.matmul"(%[[EXP]],
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
    // CHECK: %[[MM1:.*]] = "ttnn.matmul"
    // CHECK: %[[MM2:.*]] = "ttnn.matmul"
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"
    // CHECK: ttnn.d2m_subgraph @d2m_subgraph_3
    // CHECK-NEXT: ins(%[[MM1]], %[[MM2]] :
    // CHECK-NEXT: outs(%[[EMPTY]] :
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
    // CHECK: %[[MM1:.*]] = "ttnn.matmul"
    // CHECK: %[[MM2:.*]] = "ttnn.matmul"
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"
    // CHECK: ttnn.d2m_subgraph @d2m_subgraph_4
    // CHECK-NEXT: ins(%[[MM1]], %[[MM2]] :
    // CHECK-NEXT: outs(%[[EMPTY]] :
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
    // CHECK: %[[MM:.*]] = "ttnn.matmul"
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"
    // CHECK: ttnn.d2m_subgraph @d2m_subgraph_5
    // CHECK-NEXT: ins(%[[MM]] :
    // CHECK-NEXT: outs(%[[EMPTY]] :
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.neg"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.abs"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %4 = "ttnn.add"(%2, %3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x256xbf16, #layout>, tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %5 = "ttnn.log"(%4) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %5 : tensor<64x256xbf16, #layout>
  }

  // CHECK-LABEL: func.func @diamond_with_non_eltwise_consumer
  func.func @diamond_with_non_eltwise_consumer(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>, %arg2: tensor<256x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    // CHECK: "ttnn.matmul"
    // CHECK: %[[EXP:.*]] = "ttnn.exp"
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"
    // CHECK: ttnn.d2m_subgraph @d2m_subgraph_6
    // CHECK-NEXT: ins(%[[EXP]] :
    // CHECK-NEXT: outs(%[[EMPTY]] :
    // CHECK: "ttnn.matmul"(%[[EXP]],
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.neg"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.abs"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %4 = "ttnn.add"(%2, %3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x256xbf16, #layout>, tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %5 = "ttnn.log"(%4) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    // %1 exp also feeds matmul, excluding from chain; %2 %3 are now the entry ops into the chain.
    %6 = "ttnn.matmul"(%1, %arg2) : (tensor<64x256xbf16, #layout>, tensor<256x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %6 : tensor<64x256xbf16, #layout>
  }

  // CHECK-LABEL: func.func @single_eltwise_no_chain
  func.func @single_eltwise_no_chain(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.relu"
    // CHECK-NOT: ttnn.d2m_subgraph
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.relu"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %1 : tensor<64x256xbf16, #layout>
  }

  // CHECK-LABEL: func.func @two_independent_chains
  func.func @two_independent_chains(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>, %arg2: tensor<64x128xbf16, #layout>, %arg3: tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    // Chain 1: 3 ops
    // CHECK: %[[MM1:.*]] = "ttnn.matmul"
    // CHECK: %[[EMPTY1:.*]] = "ttnn.empty"
    // CHECK: ttnn.d2m_subgraph @d2m_subgraph_8
    // CHECK-NEXT: ins(%[[MM1]] :
    // CHECK-NEXT: outs(%[[EMPTY1]] :
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.log"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.neg"(%2) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    // Chain 2: 2 ops
    // CHECK: %[[MM2:.*]] = "ttnn.matmul"
    // CHECK: %[[EMPTY2:.*]] = "ttnn.empty"
    // CHECK: ttnn.d2m_subgraph @d2m_subgraph_7
    // CHECK-NEXT: ins(%[[MM2]] :
    // CHECK-NEXT: outs(%[[EMPTY2]] :
    %4 = "ttnn.matmul"(%arg2, %arg3) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %5 = "ttnn.abs"(%4) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %6 = "ttnn.sigmoid"(%5) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %6 : tensor<64x256xbf16, #layout>
  }

  // All private subgraph functions appear at the end of the module
  // CHECK: func.func private @d2m_subgraph_0(%{{.*}}: tensor<64x256xbf16
  // CHECK: "ttnn.exp"
  // CHECK: "ttnn.log"
  // CHECK: "ttnn.neg"
  // CHECK: "ttnn.abs"
  // CHECK: "ttnn.sigmoid"

  // CHECK: func.func private @d2m_subgraph_1(%[[A0:.*]]: tensor<64x256xbf16{{.*}}>, %[[A1:.*]]: tensor<64x256xbf16
  // CHECK: "ttnn.add"(%[[A0]], %[[A1]])
  // CHECK: "ttnn.exp"
  // CHECK: "ttnn.log"

  // CHECK: func.func private @d2m_subgraph_2(%{{.*}}: tensor<64x256xbf16
  // CHECK-NOT: "ttnn.exp"
  // CHECK: "ttnn.log"
  // CHECK: "ttnn.neg"

  // CHECK: func.func private @d2m_subgraph_3(%[[A0:.*]]: {{.*}}, %[[A1:.*]]:
  // CHECK: "ttnn.exp"(%[[A0]])
  // CHECK: "ttnn.neg"
  // CHECK: "ttnn.add"({{.*}}, %[[A1]])
  // CHECK: "ttnn.log"

  // CHECK: func.func private @d2m_subgraph_4(%[[A0:.*]]: {{.*}}, %[[A1:.*]]:
  // CHECK: "ttnn.exp"(%[[A0]])
  // CHECK: "ttnn.neg"(%[[A1]])
  // CHECK: "ttnn.add"
  // CHECK: "ttnn.log"

  // CHECK: func.func private @d2m_subgraph_5(%{{.*}}: tensor<64x256xbf16
  // CHECK: %[[E:.*]] = "ttnn.exp"
  // CHECK-DAG: "ttnn.neg"(%[[E]])
  // CHECK-DAG: "ttnn.abs"(%[[E]])
  // CHECK: "ttnn.add"
  // CHECK: "ttnn.log"

  // CHECK: func.func private @d2m_subgraph_6(%[[ARG:.*]]: tensor<64x256xbf16
  // exp should NOT be inside the dispatch - neg and abs use the function arg directly
  // CHECK-NOT: "ttnn.exp"
  // CHECK-DAG: "ttnn.neg"(%[[ARG]])
  // CHECK-DAG: "ttnn.abs"(%[[ARG]])
  // CHECK: "ttnn.add"
  // CHECK: "ttnn.log"

  // CHECK: func.func private @d2m_subgraph_7(%{{.*}}: tensor<64x256xbf16
  // CHECK: "ttnn.abs"
  // CHECK: "ttnn.sigmoid"

  // CHECK: func.func private @d2m_subgraph_8(%{{.*}}: tensor<64x256xbf16
  // CHECK: "ttnn.exp"
  // CHECK: "ttnn.log"
  // CHECK: "ttnn.neg"

}

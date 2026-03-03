// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttcore-wrap-device-module --ttnn-through-d2m-pipeline %s | FileCheck %s

// -----
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>

module {
  // CHECK-LABEL: func.func @one_d2m_subgraph
  func.func @one_d2m_subgraph(%arg0: tensor<64x64xbf16, #ttnn_layout>, %arg1: tensor<64x64xbf16, #ttnn_layout>, %out: tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout> {
    %0 = ttnn.d2m_subgraph @d2m_subgraph
        ins(%arg0, %arg1 : tensor<64x64xbf16, #ttnn_layout>, tensor<64x64xbf16, #ttnn_layout>)
        outs(%out : tensor<64x64xbf16, #ttnn_layout>) : tensor<64x64xbf16, #ttnn_layout>
    return %0 : tensor<64x64xbf16, #ttnn_layout>
  }
  func.func private @d2m_subgraph(%arg3: tensor<64x64xbf16, #ttnn_layout>, %arg4: tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout> {
    // CHECK: "ttnn.generic"
    // CHECK-SAME: symbol_ref = @datamovement_kernel0
    // CHECK-SAME: symbol_ref = @compute_kernel1
    %1 = "ttnn.add"(%arg3, %arg4) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #ttnn_layout>, tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout>
    return %1 : tensor<64x64xbf16, #ttnn_layout>
  }
  // CHECK: func.func private @datamovement_kernel0
  // CHECK: func.func private @compute_kernel1
}

// -----
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>
module {
  // CHECK-LABEL: func.func @two_d2m_subgraph_b2b
  func.func @two_d2m_subgraph_b2b(%arg0: tensor<64x64xbf16, #ttnn_layout>, %out0: tensor<64x64xbf16, #ttnn_layout>, %out1: tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout> {
    %0 = ttnn.d2m_subgraph @d2m_subgraph_0
        ins(%arg0 : tensor<64x64xbf16, #ttnn_layout>)
        outs(%out0 : tensor<64x64xbf16, #ttnn_layout>) : tensor<64x64xbf16, #ttnn_layout>
    %2 = ttnn.d2m_subgraph @d2m_subgraph_1
        ins(%0 : tensor<64x64xbf16, #ttnn_layout>)
        outs(%out1 : tensor<64x64xbf16, #ttnn_layout>) : tensor<64x64xbf16, #ttnn_layout>
    return %2 : tensor<64x64xbf16, #ttnn_layout>
  }
  func.func private @d2m_subgraph_0(%arg2: tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout> {
    // CHECK: "ttnn.generic"
    // CHECK-SAME: symbol_ref = @datamovement_kernel0
    // CHECK-SAME: symbol_ref = @compute_kernel1
    %1 = "ttnn.exp"(%arg2) : (tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout>
    return %1 : tensor<64x64xbf16, #ttnn_layout>
  }
  func.func private @d2m_subgraph_1(%arg3: tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout> {
    // CHECK: "ttnn.generic"
    // CHECK-SAME: symbol_ref = @datamovement_kernel2
    // CHECK-SAME: symbol_ref = @compute_kernel3
    %3 = "ttnn.log"(%arg3) : (tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout>
    return %3 : tensor<64x64xbf16, #ttnn_layout>
  }
  // CHECK: func.func private @datamovement_kernel0
  // CHECK: func.func private @compute_kernel1
  // CHECK: func.func private @datamovement_kernel2
  // CHECK: func.func private @compute_kernel3
}

// -----
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>
module {
  // CHECK-LABEL: func.func @mixed_ttnn_ops_d2m_subgraph
  func.func @mixed_ttnn_ops_d2m_subgraph(%arg0: tensor<64x64xbf16, #ttnn_layout>, %arg1: tensor<64x64xbf16, #ttnn_layout>, %out0: tensor<64x64xbf16, #ttnn_layout>, %out1: tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout> {
    // CHECK: "ttnn.add"
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #ttnn_layout>, tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout>
    %1 = ttnn.d2m_subgraph @d2m_subgraph_0
        ins(%0 : tensor<64x64xbf16, #ttnn_layout>)
        outs(%out0 : tensor<64x64xbf16, #ttnn_layout>) : tensor<64x64xbf16, #ttnn_layout>
    // CHECK: "ttnn.neg"
    %3 = "ttnn.neg"(%1) : (tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout>
    %4 = ttnn.d2m_subgraph @d2m_subgraph_1
        ins(%3 : tensor<64x64xbf16, #ttnn_layout>)
        outs(%out1 : tensor<64x64xbf16, #ttnn_layout>) : tensor<64x64xbf16, #ttnn_layout>
    return %4 : tensor<64x64xbf16, #ttnn_layout>
  }
  func.func private @d2m_subgraph_0(%arg2: tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout> {
    // CHECK: "ttnn.generic"
    // CHECK-SAME: symbol_ref = @datamovement_kernel0
    // CHECK-SAME: symbol_ref = @compute_kernel1
    %2 = "ttnn.exp"(%arg2) : (tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout>
    return %2 : tensor<64x64xbf16, #ttnn_layout>
  }
  func.func private @d2m_subgraph_1(%arg4: tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout> {
    // CHECK: "ttnn.generic"
    // CHECK-SAME: symbol_ref = @datamovement_kernel2
    // CHECK-SAME: symbol_ref = @compute_kernel3
    %5 = "ttnn.log"(%arg4) : (tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout>
    return %5 : tensor<64x64xbf16, #ttnn_layout>
  }
  // CHECK: func.func private @datamovement_kernel0
  // CHECK: func.func private @compute_kernel1
  // CHECK: func.func private @datamovement_kernel2
  // CHECK: func.func private @compute_kernel3
}

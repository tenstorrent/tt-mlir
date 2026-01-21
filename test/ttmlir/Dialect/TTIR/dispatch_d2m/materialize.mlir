// RUN: ttmlir-opt
// UNSUPPORTED: true
// ttmlir-opt --ttnn-through-d2m-pipeline test/ttmlir/Dialect/TTIR/dispatch_d2m/materialize.mlir
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>

module {
  func.func @ttnn_graph(%arg0: tensor<64x64xbf16, #ttnn_layout>, %arg1: tensor<64x64xbf16, #ttnn_layout>, %out: tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout> {
    %0 = ttnn.dispatch_d2m @d2m_subgraph
        ins(%arg0, %arg1 : tensor<64x64xbf16, #ttnn_layout>, tensor<64x64xbf16, #ttnn_layout>)
        outs(%out : tensor<64x64xbf16, #ttnn_layout>) {
      builtin.module {
        func.func @d2m_subgraph(%arg3: tensor<64x64xbf16, #ttnn_layout>, %arg4: tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout> {
          %1 = "ttnn.add"(%arg3, %arg4) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x64xbf16, #ttnn_layout>, tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout>
          return %1 : tensor<64x64xbf16, #ttnn_layout>
        }
      }
    } : tensor<64x64xbf16, #ttnn_layout>
    return %0 : tensor<64x64xbf16, #ttnn_layout>
  }
}

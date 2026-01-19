// RUN: ttmlir-opt
// UNSUPPORTED: true
// ttmlir-opt --ttnn-layout --convert-ttir-to-ttnn test/ttmlir/Dialect/TTIR/dispatch_d2m/ttir_to_ttnn.mlir

module {
  func.func @ttnn_graph(%arg0: tensor<64x64xbf16>, %arg1: tensor<64x64xbf16>, %out: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = ttir.dispatch_d2m
        ins(%arg0, %arg1 : tensor<64x64xbf16>, tensor<64x64xbf16>)
        outs(%out : tensor<64x64xbf16>) {
      func.func @d2m_subgraph(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
        %0 = "ttir.add"(%a, %b) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
        return %0 : tensor<64x64xbf16>
      }
    } : tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }
}

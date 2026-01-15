
module {
  func.func @ttnn_graph(%arg0: tensor<64x64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = "ttir.dispatch_d2m"(%arg0, %arg1) {subgraph = @d2m_subgraph} : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }

  module @d2m_subgraph {
    func.func @d2m_subgraph(%arg0: tensor<64x64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
      %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
      %1 = "ttir.subtract"(%0, %arg0) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
      %2 = "ttir.exp"(%1) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
      return %2 : tensor<64x64xbf16>
    }
  }
}

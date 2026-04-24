module {
  func.func @topk_router_gpt(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048x128xbf16>, %arg2: tensor<32x128xbf16>) -> (tensor<32x2xui16>, tensor<32x2xbf16>) {
    %0, %1 = "ttir.topk_router_gpt"(%arg0, %arg1, %arg2) <{k = 2 : i32, num_experts = 128 : i32}> : (tensor<32x2048xbf16>, tensor<2048x128xbf16>, tensor<32x128xbf16>) -> (tensor<32x2xui16>, tensor<32x2xbf16>)
    return %0, %1 : tensor<32x2xui16>, tensor<32x2xbf16>
  }
}

module {
  ttcore.device_module {
    builtin.module {
      func.func @add1(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
        %0 = ttir.empty() : tensor<32x32xf32>
        %1 = ttir.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<32x32xf32> -> tensor<32x32xf32>
        %2 = ttir.empty() : tensor<32x32xf32>
        %3 = ttir.to_layout %arg1, %2 : tensor<32x32xbf16> into tensor<32x32xf32> -> tensor<32x32xf32>
        %4 = call @hoisted_shlo_add_32x32_32x32_func_decl(%1, %3) {ttir.cpu_hoisted_call} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        %5 = ttir.empty() : tensor<32x32xbf16>
        %6 = ttir.to_layout %4, %5 : tensor<32x32xf32> into tensor<32x32xbf16> -> tensor<32x32xbf16>
        return %6 : tensor<32x32xbf16>
      }
      func.func private @hoisted_shlo_add_32x32_32x32_func_decl(tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32> attributes {ttir.cpu_hoisted_func}
    }
  }
  ttcore.cpu_module {
    builtin.module {
      func.func @hoisted_shlo_add_32x32_32x32_func(%arg0: tensor<32x32xf32> {bufferization.access = "read"}, %arg1: tensor<32x32xf32> {bufferization.access = "read"}) -> tensor<32x32xf32> attributes {arg_ranks = [2, 2], ttir.cpu_hoisted_func} {
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        return %0 : tensor<32x32xf32>
      }
    }
  }
}

module {
  ttcore.device_module {
    builtin.module {
      func.func @add1(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
        %0 = ttir.empty() : tensor<32x32xf32>
        %1 = ttir.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<32x32xf32> -> tensor<32x32xf32>
        %2 = ttir.empty() : tensor<32x32xf32>
        %3 = ttir.to_layout %arg1, %2 : tensor<32x32xbf16> into tensor<32x32xf32> -> tensor<32x32xf32>
        %4 = call @cpu_hoisted_stablehlo_add_e9d4c8b2(%1, %3) {ttir.cpu_hoisted_call} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        %5 = ttir.empty() : tensor<32x32xbf16>
        %6 = ttir.to_layout %4, %5 : tensor<32x32xf32> into tensor<32x32xbf16> -> tensor<32x32xbf16>
        return %6 : tensor<32x32xbf16>
      }
      func.func private @cpu_hoisted_stablehlo_add_e9d4c8b2(tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32> attributes {tt.function_type = "forward_cpu_declaration"}
    }
  }
  ttcore.cpu_module {
    builtin.module {
      func.func @cpu_hoisted_stablehlo_add_e9d4c8b2(%arg0: tensor<32x32xf32> {bufferization.access = "read"}, %arg1: tensor<32x32xf32> {bufferization.access = "read"}) -> tensor<32x32xf32> attributes {arg_ranks = [2, 2], tt.function_type = "forward_cpu"} {
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        return %0 : tensor<32x32xf32>
      }
    }
  }
}

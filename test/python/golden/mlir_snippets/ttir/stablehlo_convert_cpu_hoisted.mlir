module {
  ttcore.device_module {
    builtin.module {
      func.func @main(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
        %0 = ttir.empty() : tensor<32x32xf32>
        %1 = ttir.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<32x32xf32> -> tensor<32x32xf32>
        %2 = call @cpu_hoisted_stablehlo_convert_c1b2a3(%1) {ttir.cpu_hoisted_call} : (tensor<32x32xf32>) -> tensor<32x32xbf16>
        %3 = ttir.empty() : tensor<32x32xbf16>
        %4 = ttir.to_layout %2, %3 : tensor<32x32xbf16> into tensor<32x32xbf16> -> tensor<32x32xbf16>
        return %4 : tensor<32x32xbf16>
      }
      func.func private @cpu_hoisted_stablehlo_convert_c1b2a3(tensor<32x32xf32>) -> tensor<32x32xbf16> attributes {tt.function_type = "forward_cpu_declaration"}
    }
  }
  ttcore.cpu_module {
    builtin.module {
      func.func @cpu_hoisted_stablehlo_convert_c1b2a3(%arg0: tensor<32x32xf32> {bufferization.access = "read"}) -> tensor<32x32xbf16> attributes {arg_ranks = [2], tt.function_type = "forward_cpu"} {
        %0 = stablehlo.convert %arg0 : (tensor<32x32xf32>) -> tensor<32x32xbf16>
        return %0 : tensor<32x32xbf16>
      }
    }
  }
}

module {
  ttcore.cpu_module {
    builtin.module {
      func.func private @cpu_hoisted_stablehlo_dynamic_update_slice_a3f7b2c1(%arg0: tensor<1x197x768xf32> loc(unknown), %arg1: tensor<1x1x768xf32> loc(unknown), %arg2: tensor<i32> loc(unknown)) -> tensor<1x197x768xf32> attributes {tt.function_type = "forward_cpu"} {
        %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg2, %arg2 : (tensor<1x197x768xf32>, tensor<1x1x768xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x197x768xf32>
        return %0 : tensor<1x197x768xf32>
      }
    }
  }
  ttcore.device_module {
    builtin.module {
      func.func @main(%arg0: tensor<1x1x768xbf16>) -> tensor<1x197x768xbf16> {
        %6 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x197x768xbf16>}> : () -> tensor<1x197x768xbf16>
        %8 = "ttir.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
        %16 = ttir.empty() : tensor<1x197x768xf32>
        %17 = ttir.to_layout %6, %16 : tensor<1x197x768xbf16> into tensor<1x197x768xf32> -> tensor<1x197x768xf32>
        %18 = ttir.empty() : tensor<1x1x768xf32>
        %19 = ttir.to_layout %arg0, %18 : tensor<1x1x768xbf16> into tensor<1x1x768xf32> -> tensor<1x1x768xf32>
        %20 = ttir.empty() : tensor<i32>
        %21 = ttir.to_layout %8, %20 : tensor<i64> into tensor<i32> -> tensor<i32>
        %22 = call @cpu_hoisted_stablehlo_dynamic_update_slice_a3f7b2c1(%17, %19, %21) {ttir.cpu_hoisted_call} : (tensor<1x197x768xf32>, tensor<1x1x768xf32>, tensor<i32>) -> tensor<1x197x768xf32>
        %23 = ttir.empty() : tensor<1x197x768xbf16>
        %24 = ttir.to_layout %22, %23 : tensor<1x197x768xf32> into tensor<1x197x768xbf16> -> tensor<1x197x768xbf16>
        return %24 : tensor<1x197x768xbf16>
      }
    func.func private @cpu_hoisted_stablehlo_dynamic_update_slice_a3f7b2c1(tensor<1x197x768xf32>, tensor<1x1x768xf32>, tensor<i32>) -> tensor<1x197x768xf32> attributes {tt.function_type = "forward_cpu_declaration"}
    }
  }
}

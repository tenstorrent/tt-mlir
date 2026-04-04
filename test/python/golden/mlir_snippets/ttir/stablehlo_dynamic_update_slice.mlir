module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @dynamic_update_slice(%arg0: tensor<1x197x768xbf16>, %arg1: tensor<1x1x768xbf16>) -> tensor<1x197x768xbf16> {
        %0 = "ttir.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
        %1 = "ttir.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
        %2 = "ttir.reshape"(%0) <{shape = [1 : i32]}> : (tensor<i64>) -> tensor<1xi64>
        %3 = "ttir.reshape"(%1) <{shape = [1 : i32]}> : (tensor<i64>) -> tensor<1xi64>
        %4 = "ttir.concat"(%2, %3, %2) <{dim = 0 : si32}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
        %5 = "ttir.constant"() <{value = dense<0> : tensor<3xi32>}> : () -> tensor<3xi32>
        %6 = "ttir.constant"() <{value = dense<[0, 196, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
        %7 = "ttir.clamp_tensor"(%4, %5, %6) : (tensor<3xi64>, tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
        %8 = "ttir.constant"() <{value = dense<[1, 1, 768]> : tensor<3xi32>}> : () -> tensor<3xi32>
        %9 = "ttir.add"(%7, %8) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
        %10 = "ttir.slice_write"(%arg0, %arg1, %7, %9) : (tensor<1x197x768xbf16>, tensor<1x1x768xbf16>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x197x768xbf16>
        return %10 : tensor<1x197x768xbf16>
      }
    }
  }
}

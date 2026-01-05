module {
  sdy.mesh @mesh = <["x"=1, "y"=2]>
  func.func @my_modela(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"x"}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {"y"}]>] manual_axes={"x", "y"} (%arg2: tensor<32x16xf32>, %arg3: tensor<32x16xf32>) {
      %1 = stablehlo.add %arg2, %arg3 : tensor<32x16xf32>
      sdy.return %1 : tensor<32x16xf32>
    } : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}

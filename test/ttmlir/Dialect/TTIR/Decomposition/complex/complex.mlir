// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s

module {
    func.func @test_complex(%arg0: tensor<2x4xf64>, %arg1: tensor<2x4xf64>)
    -> (tensor<2x4xcomplex<f64>>) {
        %0 = "ttir.stablehlo_complex"(%arg0, %arg1) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x4xcomplex<f64>>
        return %0 : tensor<2x4xcomplex<f64>>
    }
}

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

module @jit_requantize attributes {} {
    func.func @requantize_example(%arg0: tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 0.01>> {
        %0 = stablehlo.uniform_quantize %arg0 : (tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 0.01>>
        // CHECK: = tensor.empty
        // CHECK: = "ttir.requantize"
        return %0 : tensor<1x3x224x224x!quant.uniform<i32:f32, 0.01>>
    }

    func.func @test_per_channel_requantize(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.02,0.01,0.005}>>) -> tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.01,0.005,0.0025}>> {
        %0 = stablehlo.uniform_quantize %arg0 : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.02,0.01,0.005}>>) -> tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.01,0.005,0.0025}>>
        // CHECK: = tensor.empty
        // CHECK: = "ttir.requantize"
        return %0 : tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.01,0.005,0.0025}>>
    }
}

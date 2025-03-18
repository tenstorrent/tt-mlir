// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

module @jit_requantize attributes {} {
    func.func @requantize_example(%arg0: tensor<1x3x224x224x!quant.uniform<i8:f32, 0.020787402987480164>>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 0.015625>> {
        %0 = stablehlo.uniform_quantize %arg0 : (tensor<1x3x224x224x!quant.uniform<i8:f32, 0.020787402987480164>>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 0.015625>>
        // CHECK: = tensor.empty
        // CHECK: = "ttir.requantize"
        return %0 : tensor<1x3x224x224x!quant.uniform<i8:f32, 0.015625>>
    }

    func.func @test_per_channel_requantize(%arg0: tensor<3x3x7x7x!quant.uniform<i8:f32:0, {0.020522840321063995,0.0074797165580093861,0.0056060636416077614}>>) -> tensor<3x3x7x7x!quant.uniform<i8:f32:0, {0.015625, 0.0078125, 0.00390625}>> {
        %0 = stablehlo.uniform_quantize %arg0 : (tensor<3x3x7x7x!quant.uniform<i8:f32:0, {0.020522840321063995,0.0074797165580093861,0.0056060636416077614}>>) -> tensor<3x3x7x7x!quant.uniform<i8:f32:0, {0.015625, 0.0078125, 0.00390625}>>
        // CHECK: = tensor.empty
        // CHECK: = "ttir.requantize"
        return %0 : tensor<3x3x7x7x!quant.uniform<i8:f32:0, {0.015625, 0.0078125, 0.00390625}>>
    }
}

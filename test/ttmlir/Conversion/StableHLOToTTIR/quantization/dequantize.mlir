// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_quantize attributes {} {
    func.func @dequantize_example(%arg0: tensor<1x3x224x224x!quant.uniform<i32:f32, 0.020787402987480164>>) -> tensor<1x3x224x224xf32> {
        %0 = stablehlo.uniform_dequantize %arg0 : (tensor<1x3x224x224x!quant.uniform<i32:f32, 0.020787402987480164>>) -> tensor<1x3x224x224xf32>
        // CHECK: = tensor.empty
        // CHECK: = "ttir.dequantize"
        return %0 : tensor<1x3x224x224xf32>
    }

    func.func @test_per_channel_dequantize(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.020522840321063995,0.0074797165580093861,0.0056060636416077614}>>) -> tensor<3x3x7x7xf32> {
        %0 = stablehlo.uniform_dequantize %arg0 : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.020522840321063995,0.0074797165580093861,0.0056060636416077614}>>) -> tensor<3x3x7x7xf32>
        // CHECK: = tensor.empty
        // CHECK: = "ttir.dequantize"
        return %0 : tensor<3x3x7x7xf32>
    }
}

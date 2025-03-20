// RUN: ttmlir-opt --convert-ttnn-to-emitc %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 960 + d1 * 320 + d2, d3), <1x1>, memref<30x10x!tt.tile<32x32, u8>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 960 + d1 * 320 + d2, d3), <1x1>, memref<30x10x!tt.tile<32x32, u8>, #dram>, <interleaved>>

module {
    func.func @forward(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>, #ttnn_layout>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 5.000000e-02>, #ttnn_layout1> {
        // CHECK: emitc.call_opaque "ttnn::requantize"{{.*}}
        %0 = "ttnn.requantize"(%arg0) <{
            memory_config = #ttnn.memory_config<#dram, <<30x10>>, <interleaved>>,
            output_dtype = #tt.supportedDataTypes<u8>,
            in_scale = 1.000000e-01 : f32,
            in_zero_point = 0 : i32,
            out_scale = 5.000000e-02 : f32,
            out_zero_point = 0 : i32
        }> : (tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>, #ttnn_layout>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 5.000000e-02>, #ttnn_layout1>
        return %0 : tensor<1x3x320x320x!quant.uniform<i32:f32, 5.000000e-02>, #ttnn_layout1>
    }
}

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 960 + d1 * 320 + d2, d3), <1x1>, memref<30x10x!tt.tile<32x32, u8>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 960 + d1 * 320 + d2, d3), <1x1>, memref<30x10x!tt.tile<32x32, f32>, #dram>, <interleaved>>

module {
  func.func @forward(%arg0: tensor<1x3x320x320x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x3x320x320xf32> {
    %0 = tensor.empty() : tensor<1x3x320x320xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i8:f32, 0.1>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    // CHECK: "ttnn.dequantize"
    return %1 : tensor<1x3x320x320xf32>
  }
}

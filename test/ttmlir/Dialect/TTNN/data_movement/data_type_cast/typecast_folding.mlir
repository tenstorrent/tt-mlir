// RUN: ttmlir-opt --canonicalize %s | FileCheck %s
#dram = #ttnn.buffer_type<dram>
#ttnn_layout_device_tile_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module attributes {} {
    // Test case to verify the folding of to_dtype operation.
    func.func @typecast_folding(%arg0: tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> tensor<64x128xf32, #ttnn_layout_device_tile_f32> {
        // Verify that we fold the to_dtype when we try to cast to the same dtype.
        // CHECK: return %arg0 : tensor<64x128xf32, #ttnn_layout>
        %0 = "ttnn.typecast"(%arg0) <{dtype = #tt.supportedDataTypes<f32>}> : (tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> tensor<64x128xf32, #ttnn_layout_device_tile_f32>
        return %0 : tensor<64x128xf32, #ttnn_layout_device_tile_f32>
    }
}

// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
#dram = #ttnn.buffer_type<dram>
#ttnn_layout_device_tile_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_device_tile_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_device_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {} {
    // Test case to verify the folding of typecast operation.
    func.func @typecast_folding(%arg0: tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> tensor<64x128xf32, #ttnn_layout_device_tile_f32> {
        // Verify that we fold the typecast when we try to cast to the same dtype.
        // CHECK: return %arg0 : tensor<64x128xf32, #ttnn_layout>
        %0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> tensor<64x128xf32, #ttnn_layout_device_tile_f32>
        return %0 : tensor<64x128xf32, #ttnn_layout_device_tile_f32>
    }

    // Test case to verify consecutive typecast op folding.
    func.func @typecast_folding_consecutive_typecasts(%arg0: tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16> {
        // Verify that we fold two consecutive typecast ops into a single one.
        // CHECK: ttnn.typecast
        // CHECK-NEXT: return
        %0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> tensor<64x128xi32, #ttnn_layout_device_tile_si32>
        %1 = "ttnn.typecast"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xi32, #ttnn_layout_device_tile_si32>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
        return %1 : tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
    }

    // Test case to verify that we do not fold consecutive typecast ops if the first typecast have more than a single use.
    func.func @typecast_folding_consecutive_typecasts_with_multiple_uses(%arg0: tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> (tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>, tensor<64x128xi32, #ttnn_layout_device_tile_si32>) {
        // Verify that both typecasts exists.
        // CHECK: ttnn.typecast
        // CHECK: ttnn.typecast
        %0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> tensor<64x128xi32, #ttnn_layout_device_tile_si32>
        %1 = "ttnn.typecast"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xi32, #ttnn_layout_device_tile_si32>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
        %2 = "ttnn.add"(%0, %0) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<64x128xi32, #ttnn_layout_device_tile_si32>, tensor<64x128xi32, #ttnn_layout_device_tile_si32>) -> tensor<64x128xi32, #ttnn_layout_device_tile_si32>
        return %1, %2 : tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>, tensor<64x128xi32, #ttnn_layout_device_tile_si32>
    }
}

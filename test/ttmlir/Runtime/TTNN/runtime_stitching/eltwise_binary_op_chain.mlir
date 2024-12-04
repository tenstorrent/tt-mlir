// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

// TODO: this is a workaround for compiler assuming input tensors are always on host. The ideal is to directly compile ttir graphs.
#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#system_memory = #ttnn.buffer_type<system_memory>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, bf16>, #dram>, interleaved>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #dram>, interleaved>

module attributes {tt.device = #device} {
  func.func @add(%arg0: tensor<64x128xbf16, #ttnn_layout1>, %arg1: tensor<64x128xbf16, #ttnn_layout1>) -> tensor<64x128xbf16, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xbf16, #ttnn_layout1>) -> tensor<64x128xbf16, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%arg1) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xbf16, #ttnn_layout1>) -> tensor<64x128xbf16, #ttnn_layout1>
    %3 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, #dram, <<64x128>>>, shape = #ttnn.shape<64x128>}> : (!tt.device<#device>) -> tensor<64x128xbf16, #ttnn_layout2>
    %4 = "ttnn.add"(%1, %2, %3) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16, #ttnn_layout1>, tensor<64x128xbf16, #ttnn_layout1>, tensor<64x128xbf16, #ttnn_layout2>) -> tensor<64x128xbf16, #ttnn_layout2>
    %5 = "ttnn.from_device"(%4) : (tensor<64x128xbf16, #ttnn_layout2>) -> tensor<64x128xbf16, #ttnn_layout>
    %6 = "ttnn.to_layout"(%5) <{layout = #ttnn.layout<row_major>}> : (tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout>
    return %6 : tensor<64x128xbf16, #ttnn_layout>
  }
}

module attributes {tt.device = #device} {
  func.func @multiply(%arg0: tensor<64x128xbf16, #ttnn_layout1>, %arg1: tensor<64x128xbf16, #ttnn_layout1>) -> tensor<64x128xbf16, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xbf16, #ttnn_layout1>) -> tensor<64x128xbf16, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%arg1) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xbf16, #ttnn_layout1>) -> tensor<64x128xbf16, #ttnn_layout1>
    %3 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, #dram, <<64x128>>>, shape = #ttnn.shape<64x128>}> : (!tt.device<#device>) -> tensor<64x128xbf16, #ttnn_layout2>
    %4 = "ttnn.multiply"(%1, %2, %3) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16, #ttnn_layout1>, tensor<64x128xbf16, #ttnn_layout1>, tensor<64x128xbf16, #ttnn_layout2>) -> tensor<64x128xbf16, #ttnn_layout2>
    %5 = "ttnn.from_device"(%4) : (tensor<64x128xbf16, #ttnn_layout2>) -> tensor<64x128xbf16, #ttnn_layout>
    %6 = "ttnn.to_layout"(%5) <{layout = #ttnn.layout<row_major>}> : (tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout>
    return %6 : tensor<64x128xbf16, #ttnn_layout>
  }
}

module attributes {tt.device = #device} {
  func.func @subtract(%arg0: tensor<64x128xbf16, #ttnn_layout1>, %arg1: tensor<64x128xbf16, #ttnn_layout1>) -> tensor<64x128xbf16, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xbf16, #ttnn_layout1>) -> tensor<64x128xbf16, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%arg1) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xbf16, #ttnn_layout1>) -> tensor<64x128xbf16, #ttnn_layout1>
    %3 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, #dram, <<64x128>>>, shape = #ttnn.shape<64x128>}> : (!tt.device<#device>) -> tensor<64x128xbf16, #ttnn_layout2>
    %4 = "ttnn.subtract"(%1, %2, %3) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16, #ttnn_layout1>, tensor<64x128xbf16, #ttnn_layout1>, tensor<64x128xbf16, #ttnn_layout2>) -> tensor<64x128xbf16, #ttnn_layout2>
    %5 = "ttnn.from_device"(%4) : (tensor<64x128xbf16, #ttnn_layout2>) -> tensor<64x128xbf16, #ttnn_layout>
    %6 = "ttnn.to_layout"(%5) <{layout = #ttnn.layout<row_major>}> : (tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout>
    return %6 : tensor<64x128xbf16, #ttnn_layout>
  }
}

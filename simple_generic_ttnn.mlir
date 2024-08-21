#l1_ = #tt.memory_space<l1>
#system = #tt.memory_space<system>
#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #system>>
#layout1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<8x16xf32, #system>>
module attributes {tt.device = #tt.device<#tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, [0]>, tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, physical_cores = {worker = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (1, 9), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (3, 1), (3, 2), (3, 3), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9), (4, 1), (4, 2), (4, 3), (4, 4), (4, 6), (4, 7), (4, 8), (4, 9), (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8), (5, 9), (7, 1), (7, 2), (7, 3), (7, 4), (7, 6), (7, 7), (7, 8), (7, 9), (8, 1), (8, 2), (8, 3), (8, 4), (8, 6), (8, 7), (8, 8), (8, 9), (9, 1), (9, 2), (9, 3), (9, 4), (9, 6), (9, 7), (9, 8), (9, 9)] dram = [(11, 0), (1, 0), (5, 0), (7, 0), (1, 5), (11, 5), (2, 5), (9, 5), (8, 5), (3, 5), (5, 5), (7, 5)] eth_inactive = [(6, 7), (6, 2), (6, 8), (0, 4), (0, 6), (0, 3), (0, 7), (0, 2), (0, 8), (6, 6), (0, 1), (6, 3), (0, 9)]}}], [0], [3 : i32], [<0, 0, 0, 0>]>} {
  func.func @forward(%arg0: tensor<64x128xf32, #layout>, %arg1: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout1> {
    %0 = "ttnn.open_device"() <{device_ids = [0]}> : () -> !tt.device<<#tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, [0]>>
    %1 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<<#tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, [0]>>) -> tensor<64x128xf32, #layout1>
    %2 = "ttnn.to_memory_config"(%arg0, %1) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    %3 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<<#tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, [0]>>) -> tensor<64x128xf32, #layout1>
    %4 = "ttnn.to_memory_config"(%arg1, %3) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    %5 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<<#tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, [0]>>) -> tensor<64x128xf32, #layout1>
    %6 = "ttnn.generic"(%2, %4, %5) <{circular_buffer_attributes = [#tt.circular_buffer_attributes<c_in0, <0x0, 6x6>, 4096, 2048, f32>, #tt.circular_buffer_attributes<c_in1, <0x0, 6x6>, 4096, 2048, f32>, #tt.circular_buffer_attributes<c_out0, <0x0, 6x6>, 4096, 2048, f32>], compute_attributes = [#tt.compute_attributes<<0x0, 6x6>, "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp", <hifi4, false, false, false, <1>, {ELTWISE_OP = "add_tiles", ELTWISE_OP_TYPE = "EltwiseBinaryType::ELWADD"}>>], data_movement_attributes = [#tt.data_movement_attributes<<0x0, 6x6>, "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp", <reader, <1, 1>>>, #tt.data_movement_attributes<<0x0, 6x6>, "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp", <writer, <1, 1>>>], operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    %7 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<<#tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, [0]>>) -> tensor<64x128xf32, #layout1>
    // Uncomment this line if you want to reproduce a bug
    // run command
    // ./build/bin/ttmlir-translate --ttnn-to-flatbuffer simple_generic_ttnn.mlir -o out.ttnn
    %8 = "ttnn.to_memory_config"(%6, %7) : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    "ttnn.close_device"(%0) : (!tt.device<<#tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, [0]>>) -> ()
    return %7 : tensor<64x128xf32, #layout1>
  }
}


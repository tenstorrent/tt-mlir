#l1_ = #tt.memory_space<l1>
#system = #tt.memory_space<system>
#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #system>>
#layout1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<8x16xf32, #system>>
module attributes {tt.device = #tt.device<#tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, [0]>, tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32}], [0], [<pcie|host_mmio>], [<0, 0, 0, 0>]>} {
  func.func @forward(%arg0: tensor<64x128xf32, #layout>, %arg1: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout1> {
    %0 = "ttnn.open_device"() : () -> !tt.device<<#tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, [0]>>
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


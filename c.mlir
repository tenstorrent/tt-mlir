module attributes {tt.system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1048576, num_dram_channels = 12, dram_channel_size = 1048576, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32}], [0], [<pcie|host_mmio>], [<0, 0, 0, 0>]>} {
  func.func @forward(%arg0: !emitc.opaque<"ttnn::Tensor">, %arg1: !emitc.opaque<"ttnn::Tensor">) -> !emitc.opaque<"ttnn::Tensor"> {
    %0 = emitc.call_opaque "ttnn::open_device"() : () -> !emitc.opaque<"ttnn::Device">
    %1 = emitc.call_opaque "ttnn::full"(%0) : (!emitc.opaque<"ttnn::Device">) -> !emitc.opaque<"ttnn::Tensor">
    %2 = emitc.call_opaque "ttnn::to_memory_config"(%arg0, %1) : (!emitc.opaque<"ttnn::Tensor">, !emitc.opaque<"ttnn::Tensor">) -> !emitc.opaque<"ttnn::Tensor">
    %3 = emitc.call_opaque "ttnn::full"(%0) : (!emitc.opaque<"ttnn::Device">) -> !emitc.opaque<"ttnn::Tensor">
    %4 = emitc.call_opaque "ttnn::to_memory_config"(%arg1, %3) : (!emitc.opaque<"ttnn::Tensor">, !emitc.opaque<"ttnn::Tensor">) -> !emitc.opaque<"ttnn::Tensor">
    %5 = emitc.call_opaque "ttnn::full"(%0) : (!emitc.opaque<"ttnn::Device">) -> !emitc.opaque<"ttnn::Tensor">
    %6 = emitc.call_opaque "ttnn::multiply"(%2, %4, %5) : (!emitc.opaque<"ttnn::Tensor">, !emitc.opaque<"ttnn::Tensor">, !emitc.opaque<"ttnn::Tensor">) -> !emitc.opaque<"ttnn::Tensor">
    %7 = emitc.call_opaque "ttnn::full"(%0) : (!emitc.opaque<"ttnn::Device">) -> !emitc.opaque<"ttnn::Tensor">
    %8 = emitc.call_opaque "ttnn::to_memory_config"(%6, %7) : (!emitc.opaque<"ttnn::Tensor">, !emitc.opaque<"ttnn::Tensor">) -> !emitc.opaque<"ttnn::Tensor">
    emitc.call_opaque "ttnn::close_device"(%0) : (!emitc.opaque<"ttnn::Device">) -> ()
    return %8 : !emitc.opaque<"ttnn::Tensor">
  }
}


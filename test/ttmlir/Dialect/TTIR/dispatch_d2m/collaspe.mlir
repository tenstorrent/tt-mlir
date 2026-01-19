// RUN: ttmlir-opt
// UNSUPPORTED: true
// ttmlir-opt --ttnn-collaspe-d2m test/ttmlir/Dialect/TTIR/dispatch_d2m/collaspe.mlir

#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>

module {
  func.func @ttnn_graph(%arg0: tensor<32x32xf32, #ttnn_layout>, %out: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %0 = ttnn.dispatch_d2m @d2m_entry
        ins(%arg0 : tensor<32x32xf32, #ttnn_layout>)
        outs(%out : tensor<32x32xf32, #ttnn_layout>) {
      func.func @d2m_entry(%input: tensor<32x32xf32, #ttnn_layout>, %output: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
        %dev = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %empty = "ttnn.empty"(%dev) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout>
        "ttnn.generic"(%input, %empty) <{program = #ttnn.program<kernels = [#ttnn.read_kernel<symbol_ref = @kernel0, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>], common_rt_args = [], rt_args = []>], cbs = [<total_size = 4096, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, formats = [<buffer_index = 0, dtype = f32, page_size = 4096>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>], semaphores = []>}> : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> ()
        return %empty : tensor<32x32xf32, #ttnn_layout>
      }
      func.func private @kernel0() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
        return
      }
    } : tensor<32x32xf32, #ttnn_layout>
    return %0 : tensor<32x32xf32, #ttnn_layout>
  }

  // CHECK: func.func @d2m_entry_kernel0
}

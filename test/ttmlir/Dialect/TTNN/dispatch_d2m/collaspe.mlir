// RUN: ttmlir-opt --split-input-file --ttnn-collaspe-d2m %s | FileCheck %s

#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>

module {
  // CHECK-LABEL: func.func @one_d2m_subgraph
  func.func @one_d2m_subgraph(%arg0: tensor<32x32xf32, #ttnn_layout>, %out: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    // CHECK-NOT: ttnn.d2m_subgraph
    // CHECK: "ttnn.generic"
    %0 = ttnn.d2m_subgraph @d2m_subgraph_0
        ins(%arg0 : tensor<32x32xf32, #ttnn_layout>)
        outs(%out : tensor<32x32xf32, #ttnn_layout>) : tensor<32x32xf32, #ttnn_layout>
    return %0 : tensor<32x32xf32, #ttnn_layout>
  }
  // CHECK-NOT: func.func private @d2m_subgraph_0
  func.func private @d2m_subgraph_0(%input: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %empty = "ttnn.empty"(%device) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <block_sharded>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout>
    "ttnn.generic"(%input, %empty) <{program = #ttnn.program<kernels = [#ttnn.read_kernel<symbol_ref = @kernel0, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>], common_rt_args = [], rt_args = []>], cbs = [<total_size = 4096, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, formats = [<buffer_index = 0, dtype = f32, page_size = 4096>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>], semaphores = []>}> : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> ()
    return %empty : tensor<32x32xf32, #ttnn_layout>
  }
  func.func private @kernel0() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  // CHECK: func.func private @kernel0
}

// -----
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>

module {
  // CHECK-LABEL: func.func @two_d2m_subgraph_b2b
  func.func @two_d2m_subgraph_b2b(%arg0: tensor<32x32xf32, #ttnn_layout>, %out0: tensor<32x32xf32, #ttnn_layout>, %out1: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    // CHECK-NOT: ttnn.d2m_subgraph
    // CHECK: "ttnn.generic"
    %0 = ttnn.d2m_subgraph @d2m_subgraph_0
        ins(%arg0 : tensor<32x32xf32, #ttnn_layout>)
        outs(%out0 : tensor<32x32xf32, #ttnn_layout>) : tensor<32x32xf32, #ttnn_layout>
    // CHECK-NOT: ttnn.d2m_subgraph
    // CHECK: "ttnn.generic"
    %2 = ttnn.d2m_subgraph @d2m_subgraph_1
        ins(%0 : tensor<32x32xf32, #ttnn_layout>)
        outs(%out1 : tensor<32x32xf32, #ttnn_layout>) : tensor<32x32xf32, #ttnn_layout>
    return %2 : tensor<32x32xf32, #ttnn_layout>
  }
  func.func private @d2m_subgraph_0(%input: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %empty = "ttnn.empty"(%device) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <block_sharded>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout>
    "ttnn.generic"(%input, %empty) <{program = #ttnn.program<kernels = [#ttnn.read_kernel<symbol_ref = @kernel1, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>], common_rt_args = [], rt_args = []>], cbs = [<total_size = 4096, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, formats = [<buffer_index = 0, dtype = f32, page_size = 4096>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>], semaphores = []>}> : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> ()
    return %empty : tensor<32x32xf32, #ttnn_layout>
  }
  func.func private @d2m_subgraph_1(%input: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %empty = "ttnn.empty"(%device) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <block_sharded>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout>
    "ttnn.generic"(%input, %empty) <{program = #ttnn.program<kernels = [#ttnn.read_kernel<symbol_ref = @kernel2, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>], common_rt_args = [], rt_args = []>], cbs = [<total_size = 4096, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, formats = [<buffer_index = 0, dtype = f32, page_size = 4096>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>], semaphores = []>}> : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> ()
    return %empty : tensor<32x32xf32, #ttnn_layout>
  }
  func.func private @kernel1() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @kernel2() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  // CHECK: func.func private @kernel1
  // CHECK: func.func private @kernel2
}

// -----
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>

module {
  // CHECK-LABEL: func.func @mixed_ttnn_ops_d2m_subgraph
  func.func @mixed_ttnn_ops_d2m_subgraph(%arg0: tensor<32x32xf32, #ttnn_layout>, %arg1: tensor<32x32xf32, #ttnn_layout>, %out0: tensor<32x32xf32, #ttnn_layout>, %out1: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    // CHECK: "ttnn.add"
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout>
    // CHECK-NOT: ttnn.d2m_subgraph
    // CHECK: "ttnn.generic"
    %1 = ttnn.d2m_subgraph @d2m_subgraph_0
        ins(%0 : tensor<32x32xf32, #ttnn_layout>)
        outs(%out0 : tensor<32x32xf32, #ttnn_layout>) : tensor<32x32xf32, #ttnn_layout>
    // CHECK: "ttnn.exp"
    %2 = "ttnn.exp"(%1) : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout>
    // CHECK-NOT: ttnn.d2m_subgraph
    // CHECK: "ttnn.generic"
    %3 = ttnn.d2m_subgraph @d2m_subgraph_1
        ins(%2 : tensor<32x32xf32, #ttnn_layout>)
        outs(%out1 : tensor<32x32xf32, #ttnn_layout>) : tensor<32x32xf32, #ttnn_layout>
    return %3 : tensor<32x32xf32, #ttnn_layout>
  }
  func.func private @d2m_subgraph_0(%input: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %empty = "ttnn.empty"(%device) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <block_sharded>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout>
    "ttnn.generic"(%input, %empty) <{program = #ttnn.program<kernels = [#ttnn.read_kernel<symbol_ref = @kernel3, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>], common_rt_args = [], rt_args = []>], cbs = [<total_size = 4096, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, formats = [<buffer_index = 0, dtype = f32, page_size = 4096>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>], semaphores = []>}> : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> ()
    return %empty : tensor<32x32xf32, #ttnn_layout>
  }
  func.func private @d2m_subgraph_1(%input: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %empty = "ttnn.empty"(%device) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <block_sharded>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout>
    "ttnn.generic"(%input, %empty) <{program = #ttnn.program<kernels = [#ttnn.read_kernel<symbol_ref = @kernel4, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>], common_rt_args = [], rt_args = []>], cbs = [<total_size = 4096, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, formats = [<buffer_index = 0, dtype = f32, page_size = 4096>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>], semaphores = []>}> : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> ()
    return %empty : tensor<32x32xf32, #ttnn_layout>
  }
  func.func private @kernel3() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @kernel4() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  // CHECK: func.func private @kernel3
  // CHECK: func.func private @kernel4
}

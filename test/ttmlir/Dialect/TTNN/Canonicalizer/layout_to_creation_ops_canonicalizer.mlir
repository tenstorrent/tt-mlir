// RUN: ttmlir-opt --ttcore-register-device --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test cases to verify that the to_layout op merges correctly with creation ops.
// Creation ops:
// Empty Op, Arange Op, Rand Op, Full Op, Constant Op, Zeros Op, Ones Op

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_host_f32_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #system_memory>>
#ttnn_layout_dram_f32_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #dram>, <interleaved>>
#ttnn_layout_dram_bf16_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_dram_bf16_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #dram>, <interleaved>>
#ttnn_layout_host_bf16_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>
#ttnn_layout_1_host_f32_rm = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<32xf32, #system_memory>>
#ttnn_layout_1_device_bf16_tile = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {} {
  // Verify that to_layout op merges into empty op.
  func.func @to_layout_merge_into_empty_on_device() -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout_dram_f32_rm>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout_dram_f32_rm>) -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
    return %2 : tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
  }

  //Verify that to_layout op doesn't merge into empty op if to layout moves the tensor to host.
  func.func @to_layout_not_merge_into_empty_from_device_to_host() -> tensor<32x32xbf16, #ttnn_layout_host_bf16_tile> {
    // CHECK: "ttnn.zeros"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout_dram_f32_rm>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xf32, #ttnn_layout_dram_f32_rm>) -> tensor<32x32xbf16, #ttnn_layout_host_bf16_tile>
    return %2 : tensor<32x32xbf16, #ttnn_layout_host_bf16_tile>
  }

  // Verify that to_layout op merges into empty op.
  func.func @to_layout_merge_into_rand_on_device() -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile> {
    // CHECK: "ttnn.rand"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.rand"(%0) <{size = #ttnn.shape<32x32>, dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout_dram_f32_rm>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout_dram_f32_rm>) -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
    return %2 : tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
  }

  //Verify that to_layout op doesn't merge into empty op if to layout moves the tensor to host.
  func.func @to_layout_not_merge_into_rand_from_device_to_host() -> tensor<32x32xbf16, #ttnn_layout_host_bf16_tile> {
    // CHECK: "ttnn.rand"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-NOT: "ttnn.to_layout"
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.rand"(%0) <{size = #ttnn.shape<32x32>, dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout_dram_f32_rm>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xf32, #ttnn_layout_dram_f32_rm>) -> tensor<32x32xbf16, #ttnn_layout_host_bf16_tile>
    return %2 : tensor<32x32xbf16, #ttnn_layout_host_bf16_tile>
  }

  // Verify that to_layout op merges into arange op.
  func.func @to_layout_merge_into_arange_from_host_to_device() -> tensor<32xbf16, #ttnn_layout_1_device_bf16_tile> {
    // CHECK: "ttnn.arange"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.arange"() <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>, start = 0 : i64, step = 1 : i64, end = 32 : i64}> : () -> tensor<32xf32, #ttnn_layout_1_host_f32_rm>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32xf32, #ttnn_layout_1_host_f32_rm>) -> tensor<32xbf16, #ttnn_layout_1_device_bf16_tile>
    return %2 : tensor<32xbf16, #ttnn_layout_1_device_bf16_tile>
  }

  // Verify that to_layout op merges into arange op.
  func.func @to_layout_merge_into_arange_from_device_to_host() -> tensor<32xf32, #ttnn_layout_1_host_f32_rm> {
    // CHECK: "ttnn.arange"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.arange"() <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, start = 0 : i64, step = 1 : i64, end = 32 : i64}> : () -> tensor<32xf32, #ttnn_layout_1_device_bf16_tile>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32xf32, #ttnn_layout_1_device_bf16_tile>) -> tensor<32xf32, #ttnn_layout_1_host_f32_rm>
    return %2 : tensor<32xf32, #ttnn_layout_1_host_f32_rm>
  }

  // Verify that to_layout op merges into zeros op.
  func.func @to_layout_merge_into_zeros_from_host_to_device() -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile> {
    // CHECK: "ttnn.zeros"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.zeros"() <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>, shape = #ttnn.shape<32x32>}> : () -> tensor<32x32xf32, #ttnn_layout_host_f32_rm>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout_host_f32_rm>) -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
    return %2 : tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
  }

  // Verify that to_layout op merges into zeros op.
  func.func @to_layout_merge_into_zeros_from_device_to_host() -> tensor<32x32xf32, #ttnn_layout_host_f32_rm> {
    // CHECK: "ttnn.zeros"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.zeros"() <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : () -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>) -> tensor<32x32xf32, #ttnn_layout_host_f32_rm>
    return %2 : tensor<32x32xf32, #ttnn_layout_host_f32_rm>
  }

  // Verify that to_layout op merges into ones op.
  func.func @to_layout_merge_into_ones_from_host_to_device() -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile> {
    // CHECK: "ttnn.ones"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.ones"() <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>, shape = #ttnn.shape<32x32>}> : () -> tensor<32x32xf32, #ttnn_layout_host_f32_rm>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout_host_f32_rm>) -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
    return %2 : tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
  }

  // Verify that to_layout op merges into ones op.
  func.func @to_layout_merge_into_ones_from_device_to_host() -> tensor<32x32xf32, #ttnn_layout_host_f32_rm> {
    // CHECK: "ttnn.ones"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.ones"() <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : () -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>) -> tensor<32x32xf32, #ttnn_layout_host_f32_rm>
    return %2 : tensor<32x32xf32, #ttnn_layout_host_f32_rm>
  }

  // Verify that to_layout op merges into ones op.
  func.func @to_layout_merge_into_full_from_host_to_device() -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile> {
    // CHECK: "ttnn.full"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.full"() <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>, shape = #ttnn.shape<32x32>, fill_value = 7.0 : f32}> : () -> tensor<32x32xf32, #ttnn_layout_host_f32_rm>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout_host_f32_rm>) -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
    return %2 : tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
  }

  // Verify that to_layout op merges into full op.
  func.func @to_layout_merge_into_full_from_device_to_host() -> tensor<32x32xf32, #ttnn_layout_host_f32_rm> {
    // CHECK: "ttnn.full"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.full"() <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>, fill_value = 7.0 : f32}> : () -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>) -> tensor<32x32xf32, #ttnn_layout_host_f32_rm>
    return %2 : tensor<32x32xf32, #ttnn_layout_host_f32_rm>
  }

  // Verify that to_layout op merges into constant op.
  func.func @to_layout_merge_into_constant_from_host_to_device() -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.constant"() <{value = dense_resource<dense_attr_f32> : tensor<32x32xf32>, dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>, shape = #ttnn.shape<32x32>, fill_value = 7.0 : f32}> : () -> tensor<32x32xf32, #ttnn_layout_host_f32_rm>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout_host_f32_rm>) -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
    return %2 : tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
  }

  // Verify that to_layout op merges into constant op.
  func.func @to_layout_merge_into_constant_from_device_to_host() -> tensor<32x32xf32, #ttnn_layout_host_f32_rm> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    // CHECK-NOT: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.constant"(%0) <{value = dense_resource<dense_attr_bf16> : tensor<32x32xbf16>, dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>, fill_value = 7.0 : f32}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>) -> tensor<32x32xf32, #ttnn_layout_host_f32_rm>
    return %2 : tensor<32x32xf32, #ttnn_layout_host_f32_rm>
  }

  // Verify that the canonicalization shouldn't happen if the creation op has more than one use.
  func.func @to_layout_not_merge_into_constant_with_more_than_one_use() -> (tensor<32x32xf32, #ttnn_layout_host_f32_rm>, tensor<32x32xf32, #ttnn_layout_dram_bf16_rm>) {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK: "ttnn.to_layout"
    // CHECK: "ttnn.to_layout"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.constant"(%0) <{value = dense_resource<dense_attr_bf16> : tensor<32x32xbf16>, dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>, fill_value = 7.0 : f32}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>) -> tensor<32x32xf32, #ttnn_layout_host_f32_rm>
    %3 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout_dram_bf16_tile>) -> tensor<32x32xf32, #ttnn_layout_dram_bf16_rm>
    return %2, %3 : tensor<32x32xf32, #ttnn_layout_host_f32_rm>, tensor<32x32xf32, #ttnn_layout_dram_bf16_rm>
  }
}

{-#
    dialect_resources: {
        builtin: {
            // This should encode for 32x32xbf16 values which are both 2.0
            // 0x020000000 is a hex string blob
            // 512 values 0x003F is 1.0 in bfloat16
            // 512 values 0x0040 is 2.0 in bfloat16
            dense_attr_bf16: "0x02000000803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040803f0040",
            dense_attr_f32: "0x040000000000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f000000400000803f00000040"
        }
    }
#-}

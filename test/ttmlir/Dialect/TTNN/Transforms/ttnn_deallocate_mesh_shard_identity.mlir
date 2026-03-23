// RUN: ttmlir-opt --ttnn-deallocate -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>

#host = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64xbf16, #system_memory>>
#full = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#shard = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#perm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  func.func @forward(%arg0: tensor<64x64xbf16, #host>) -> tensor<64x32xbf16, #perm> {
    %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %0 = "ttnn.to_device"(%arg0, %dev) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x64xbf16, #host>, !ttnn.device) -> tensor<64x64xbf16, #full>
    %1 = "ttnn.mesh_shard"(%0, %dev) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<64x64xbf16, #full>, !ttnn.device) -> tensor<32x64xbf16, #shard>
    %2 = "ttnn.permute"(%1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16, #shard>) -> tensor<64x32xbf16, #perm>
    return %2 : tensor<64x32xbf16, #perm>
  }
}

// CHECK-LABEL: func.func @forward
// CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"
// CHECK: %[[MESH:.*]] = "ttnn.mesh_shard"(%[[TO_DEV]]
// CHECK: %[[PERM:.*]] = "ttnn.permute"(%[[MESH]])
// CHECK-NOT: "ttnn.deallocate"(%[[TO_DEV]])
// CHECK: "ttnn.deallocate"(%[[TO_DEV]])
// CHECK-NOT: "ttnn.deallocate"(%[[MESH]])

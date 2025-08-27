// UNSUPPORTED: true
// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path% mesh-shape=2,4" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 256 + d2, d3), <1x1>, memref<256x512xf32, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<128x128xf32, #system_memory>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #system_memory>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @point_to_point_test attributes {} {
  func.func public @point_to_point_provide_output_tensor(%arg0: tensor<1x1x256x512xf32, #ttnn_layout1>) -> (tensor<1x1x256x512xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
    %1 = "ttnn.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32, #ttnn_layout1>, !ttnn.device) -> tensor<1x1x128x128xf32, #ttnn_layout2>
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x256x512xf32, #ttnn_layout1>) -> ()
    %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<1x1x128x128xf32, #ttnn_layout2>) -> tensor<1x1x128x128xf32, #ttnn_layout3>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout2>) -> ()
    %3 = "ttnn.to_device"(%2, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x1x128x128xf32, #ttnn_layout3>, !ttnn.device) -> tensor<1x1x128x128xf32, #ttnn_layout4>
    "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout3>) -> ()
    %4 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<1x1x128x128>}> : (!ttnn.device) -> tensor<1x1x128x128xf32, #ttnn_layout4>
    %5 = "ttnn.point_to_point"(%3, %4) <{receive_coord = array<i64: 0, 1>, send_coord = array<i64: 0, 0>}> : (tensor<1x1x128x128xf32, #ttnn_layout4>, tensor<1x1x128x128xf32, #ttnn_layout4>) -> tensor<1x1x128x128xf32, #ttnn_layout4>
    "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout4>) -> ()
    %6 = "ttnn.point_to_point"(%3, %5) <{receive_coord = array<i64: 0, 0>, send_coord = array<i64: 0, 1>}> : (tensor<1x1x128x128xf32, #ttnn_layout4>, tensor<1x1x128x128xf32, #ttnn_layout4>) -> tensor<1x1x128x128xf32, #ttnn_layout4>
    "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout4>) -> ()
    %7 = "ttnn.from_device"(%6) : (tensor<1x1x128x128xf32, #ttnn_layout4>) -> tensor<1x1x128x128xf32, #ttnn_layout3>
    "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout4>) -> ()
    %8 = "ttnn.to_layout"(%7) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1x128x128xf32, #ttnn_layout3>) -> tensor<1x1x128x128xf32, #ttnn_layout2>
    "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout3>) -> ()
    %9 = "ttnn.mesh_shard"(%8, %0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x128x128xf32, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x256x512xf32, #ttnn_layout1>
    "ttnn.deallocate"(%8) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout2>) -> ()
    return %9 : tensor<1x1x256x512xf32, #ttnn_layout1>
  }
  func.func public @point_to_point_case_generate_output_tensor(%arg0: tensor<1x1x256x512xf32, #ttnn_layout1>) -> (tensor<1x1x256x512xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
    %1 = "ttnn.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32, #ttnn_layout1>, !ttnn.device) -> tensor<1x1x128x128xf32, #ttnn_layout2>
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x256x512xf32, #ttnn_layout1>) -> ()
    %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<1x1x128x128xf32, #ttnn_layout2>) -> tensor<1x1x128x128xf32, #ttnn_layout3>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout2>) -> ()
    %3 = "ttnn.to_device"(%2, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x1x128x128xf32, #ttnn_layout3>, !ttnn.device) -> tensor<1x1x128x128xf32, #ttnn_layout4>
    "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout3>) -> ()
    %5 = "ttnn.point_to_point"(%3) <{receive_coord = array<i64: 0, 1>, send_coord = array<i64: 0, 0>}> : (tensor<1x1x128x128xf32, #ttnn_layout4>) -> tensor<1x1x128x128xf32, #ttnn_layout4>
    %6 = "ttnn.point_to_point"(%3, %5) <{receive_coord = array<i64: 0, 0>, send_coord = array<i64: 0, 1>}> : (tensor<1x1x128x128xf32, #ttnn_layout4>, tensor<1x1x128x128xf32, #ttnn_layout4>) -> tensor<1x1x128x128xf32, #ttnn_layout4>
    "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout4>) -> ()
    %7 = "ttnn.from_device"(%6) : (tensor<1x1x128x128xf32, #ttnn_layout4>) -> tensor<1x1x128x128xf32, #ttnn_layout3>
    "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout4>) -> ()
    %8 = "ttnn.to_layout"(%7) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1x128x128xf32, #ttnn_layout3>) -> tensor<1x1x128x128xf32, #ttnn_layout2>
    "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout3>) -> ()
    %9 = "ttnn.mesh_shard"(%8, %0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x128x128xf32, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x256x512xf32, #ttnn_layout1>
    "ttnn.deallocate"(%8) <{force = false}> : (tensor<1x1x128x128xf32, #ttnn_layout2>) -> ()
    return %9 : tensor<1x1x256x512xf32, #ttnn_layout1>
  }
}

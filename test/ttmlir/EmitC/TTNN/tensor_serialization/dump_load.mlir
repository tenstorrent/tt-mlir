// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

#system_memory = #ttnn.buffer_type<system_memory>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout_host = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #system_memory>>
#ttnn_layout_device = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  // Test dump and load operations with tensor on device.
  func.func @dump_load_device_tensor(%arg0: tensor<32x32xf32, #ttnn_layout_device>) -> tensor<32x32xf32, #ttnn_layout_device> {
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    "ttnn.dump_tensor"(%arg0) <{file_path = "device_tensor.tensorbin"}> : (tensor<32x32xf32, #ttnn_layout_device>) -> ()
    %loaded = "ttnn.load_tensor"(%device) <{file_path = "device_tensor.tensorbin"}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout_device>
    return %loaded : tensor<32x32xf32, #ttnn_layout_device>
  }

  // Test dump and load operations with tensor in system memory.
  func.func @dump_load_system_tensor(%arg0: tensor<32x32xf32, #ttnn_layout_host>) -> tensor<32x32xf32, #ttnn_layout_host> {
    "ttnn.dump_tensor"(%arg0) <{file_path = "system_tensor.tensorbin"}> : (tensor<32x32xf32, #ttnn_layout_host>) -> ()
    %loaded = "ttnn.load_tensor"() <{file_path = "system_tensor.tensorbin"}> : () -> tensor<32x32xf32, #ttnn_layout_host>
    return %loaded : tensor<32x32xf32, #ttnn_layout_host>
  }
}

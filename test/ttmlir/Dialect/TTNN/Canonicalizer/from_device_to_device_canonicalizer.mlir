// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#host_tiled = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>
#device_tiled = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @fold_from_device_to_device(%arg0: tensor<32x32xbf16, #host_tiled>) -> tensor<32x32xbf16, #host_tiled> {
    // CHECK-LABEL: func.func @fold_from_device_to_device
    // CHECK-NOT: "ttnn.to_device"
    // CHECK-NOT: "ttnn.from_device"
    // CHECK: return %arg0
    %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %0 = "ttnn.to_device"(%arg0, %dev) : (tensor<32x32xbf16, #host_tiled>, !ttnn.device) -> tensor<32x32xbf16, #device_tiled>
    %1 = "ttnn.from_device"(%0) : (tensor<32x32xbf16, #device_tiled>) -> tensor<32x32xbf16, #host_tiled>
    return %1 : tensor<32x32xbf16, #host_tiled>
  }
}

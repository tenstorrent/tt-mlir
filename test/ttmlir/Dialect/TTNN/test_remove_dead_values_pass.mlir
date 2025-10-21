// RUN: ttmlir-opt --remove-dead-values -o %t %s
// RUN: FileCheck %s --input-file=%t
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #dram>, <interleaved>>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout1>
    %2 = "ttnn.to_device"(%1, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout1>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout1>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    %3 = "ttnn.to_layout"(%arg1) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout1>
    %4 = "ttnn.to_device"(%3, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout1>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout1>
    "ttnn.deallocate"(%3) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    // CHECK: "ttnn.multiply"
    %5 = "ttnn.multiply"(%2, %4) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x128xf32, #ttnn_layout1>, tensor<64x128xf32, #ttnn_layout1>) -> tensor<64x128xf32, #ttnn_layout2>
    "ttnn.deallocate"(%4) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    "ttnn.deallocate"(%2) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    %6 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout1>
    %7 = "ttnn.to_device"(%6, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout1>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout1>
    "ttnn.deallocate"(%6) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    %8 = "ttnn.to_layout"(%arg1) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout1>
    %9 = "ttnn.to_device"(%8, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout1>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout1>
    "ttnn.deallocate"(%8) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    // CHECK-NOT: "ttnn.add"
    %10 = "ttnn.add"(%7, %9) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x128xf32, #ttnn_layout1>, tensor<64x128xf32, #ttnn_layout1>) -> tensor<64x128xf32, #ttnn_layout2>
    "ttnn.deallocate"(%9) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    "ttnn.deallocate"(%7) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    %11 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout1>
    %12 = "ttnn.to_device"(%11, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout1>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout1>
    "ttnn.deallocate"(%11) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    %13 = "ttnn.to_layout"(%arg1) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout1>
    %14 = "ttnn.to_device"(%13, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout1>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout1>
    "ttnn.deallocate"(%13) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    // CHECK-NOT: "ttnn.subtract"
    %15 = "ttnn.subtract"(%12, %14) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x128xf32, #ttnn_layout1>, tensor<64x128xf32, #ttnn_layout1>) -> tensor<64x128xf32, #ttnn_layout2>
    "ttnn.deallocate"(%14) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    "ttnn.deallocate"(%12) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    %16 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout1>
    %17 = "ttnn.to_device"(%16, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout1>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout1>
    "ttnn.deallocate"(%16) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    %18 = "ttnn.to_layout"(%arg1) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout1>
    %19 = "ttnn.to_device"(%18, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout1>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout1>
    "ttnn.deallocate"(%18) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    // CHECK-NOT: "ttnn.divide"
    %20 = "ttnn.divide"(%17, %19) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x128xf32, #ttnn_layout1>, tensor<64x128xf32, #ttnn_layout1>) -> tensor<64x128xf32, #ttnn_layout2>
    "ttnn.deallocate"(%19) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    "ttnn.deallocate"(%17) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    %21 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout1>
    %22 = "ttnn.to_device"(%21, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout1>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout1>
    "ttnn.deallocate"(%21) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    %23 = "ttnn.to_layout"(%arg1) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout1>
    %24 = "ttnn.to_device"(%23, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout1>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout1>
    "ttnn.deallocate"(%23) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    // CHECK-NOT: "ttnn.eq"
    %25 = "ttnn.eq"(%22, %24)<{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x128xf32, #ttnn_layout1>, tensor<64x128xf32, #ttnn_layout1>) -> tensor<64x128xf32, #ttnn_layout2>
    "ttnn.deallocate"(%24) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    "ttnn.deallocate"(%22) <{force = false}> : (tensor<64x128xf32, #ttnn_layout1>) -> ()
    %26 = "ttnn.from_device"(%5) : (tensor<64x128xf32, #ttnn_layout2>) -> tensor<64x128xf32, #ttnn_layout>
    return %26 : tensor<64x128xf32, #ttnn_layout>
  }
}

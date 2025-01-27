// RUN: ttmlir-opt --ttnn-decompose-layouts %s | FileCheck %s
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #system_memory>>
module {
    func.func @data_cast_on_host(%arg0: tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout1> {
        // Check that the input operand is casted to bf16 on host
        // CHECK: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
        %1 = "ttnn.to_layout"(%arg0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory, <<64x128>>>}> : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout1>
        return %1 : tensor<64x128xbf16, #ttnn_layout1>
    }
}

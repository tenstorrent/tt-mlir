// RUN: ttmlir-opt --convert-ttnn-to-ttir -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>

module {
    func.func @test_full() -> tensor<32x32xbf16, #ttnn_layout> {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

        // CHECK: %{{[0-9]+}} = "ttir.full"() <{fill_value = 5.000000e-01 : f32, shape = array<i32: 32, 32>}>
        %2 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 5.000000e-01 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> {ttnn.hoist_generic_via_d2m} : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>

        return %2 : tensor<32x32xbf16, #ttnn_layout>
    }
}

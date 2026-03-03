// RUN: ttmlir-opt --ttir-complex-data-type-conversion %s

func.func @main(%arg0: tensor<16x8xcomplex<f32>> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<16x8xcomplex<f32>>>, ttir.name = "args_1"}, %arg1: tensor<2x16x4x16xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<2x16x4x16xbf16>>, ttir.name = "args_0"}) -> (tensor<2x16x4x16xbf16> {ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<2x16x4x16xbf16>>}) attributes {tt.function_type = "forward_device"} {
    %0 = "ttir.typecast"(%arg1) <{conservative_folding = false}> : (tensor<2x16x4x16xbf16>) -> tensor<2x16x4x16xf32>
    %1 = "ttir.reshape"(%0) <{shape = [2 : i32, 16 : i32, 4 : i32, 8 : i32, 2 : i32]}> : (tensor<2x16x4x16xf32>) -> tensor<2x16x4x8x2xf32>
    %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 16 : i32, 4 : i32, 8 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x16x4x8x2xf32>) -> tensor<2x16x4x8x1xf32>
    %3 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [2 : i32, 16 : i32, 4 : i32, 8 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x16x4x8x2xf32>) -> tensor<2x16x4x8x1xf32>
    %4 = "ttir.reshape"(%2) <{shape = [2 : i32, 16 : i32, 4 : i32, 8 : i32]}> : (tensor<2x16x4x8x1xf32>) -> tensor<2x16x4x8xf32>
    %5 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 16 : i32, 1 : i32, 8 : i32]}> : (tensor<16x8xcomplex<f32>>) -> tensor<1x16x1x8xcomplex<f32>>
    %6 = "ttir.stablehlo_real"(%5) : (tensor<1x16x1x8xcomplex<f32>>) -> tensor<1x16x1x8xf32>
    %7 = "ttir.broadcast"(%6) <{broadcast_dimensions = array<i64: 2, 1, 4, 1>}> : (tensor<1x16x1x8xf32>) -> tensor<2x16x4x8xf32>
    %8 = "ttir.multiply"(%4, %7) : (tensor<2x16x4x8xf32>, tensor<2x16x4x8xf32>) -> tensor<2x16x4x8xf32>
    %9 = "ttir.reshape"(%3) <{shape = [2 : i32, 16 : i32, 4 : i32, 8 : i32]}> : (tensor<2x16x4x8x1xf32>) -> tensor<2x16x4x8xf32>
    %10 = "ttir.stablehlo_imag"(%5) : (tensor<1x16x1x8xcomplex<f32>>) -> tensor<1x16x1x8xf32>
    %11 = "ttir.broadcast"(%10) <{broadcast_dimensions = array<i64: 2, 1, 4, 1>}> : (tensor<1x16x1x8xf32>) -> tensor<2x16x4x8xf32>
    %12 = "ttir.multiply"(%9, %11) : (tensor<2x16x4x8xf32>, tensor<2x16x4x8xf32>) -> tensor<2x16x4x8xf32>
    %13 = "ttir.subtract"(%8, %12) : (tensor<2x16x4x8xf32>, tensor<2x16x4x8xf32>) -> tensor<2x16x4x8xf32>
    %14 = "ttir.multiply"(%4, %11) : (tensor<2x16x4x8xf32>, tensor<2x16x4x8xf32>) -> tensor<2x16x4x8xf32>
    %15 = "ttir.multiply"(%9, %7) : (tensor<2x16x4x8xf32>, tensor<2x16x4x8xf32>) -> tensor<2x16x4x8xf32>
    %16 = "ttir.add"(%14, %15) : (tensor<2x16x4x8xf32>, tensor<2x16x4x8xf32>) -> tensor<2x16x4x8xf32>
    %17 = "ttir.reshape"(%13) <{shape = [2 : i32, 16 : i32, 4 : i32, 8 : i32, 1 : i32]}> : (tensor<2x16x4x8xf32>) -> tensor<2x16x4x8x1xf32>
    %18 = "ttir.reshape"(%16) <{shape = [2 : i32, 16 : i32, 4 : i32, 8 : i32, 1 : i32]}> : (tensor<2x16x4x8xf32>) -> tensor<2x16x4x8x1xf32>
    %19 = "ttir.concat"(%17, %18) <{dim = 4 : si32}> : (tensor<2x16x4x8x1xf32>, tensor<2x16x4x8x1xf32>) -> tensor<2x16x4x8x2xf32>
    %20 = "ttir.reshape"(%19) <{shape = [2 : i32, 16 : i32, 4 : i32, 16 : i32]}> : (tensor<2x16x4x8x2xf32>) -> tensor<2x16x4x16xf32>
    %21 = "ttir.typecast"(%20) <{conservative_folding = false}> : (tensor<2x16x4x16xf32>) -> tensor<2x16x4x16xbf16>
    return %21 : tensor<2x16x4x16xbf16>
}

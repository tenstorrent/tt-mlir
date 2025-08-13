module @SyncTensorsGraph.337 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x8>]>} {
  func.func @main(%arg0: tensor<1x14xi64> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg1: tensor<128256x4096xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg2: tensor<14xi64> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg3: tensor<64xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg4: tensor<1024x4096xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg5: tensor<f32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg6: tensor<4096xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg7: tensor<1024x4096xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg8: tensor<4096x14336xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg9: tensor<14336x4096xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg10: tensor<4096x4096xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg11: tensor<f32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg12: tensor<f32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg13: tensor<4096x4096xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg14: tensor<4096xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg15: tensor<14336x4096xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg16: tensor<4096xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg17: tensor<128256x4096xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<1x14x4096xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x19x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x19x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x14x4096xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<14x128256xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x14x128256xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = ttir.empty() : tensor<1x14xi64>
    %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x14xi64>, tensor<1x14xi64>) -> tensor<1x14xi64>
    %2 = ttir.empty() : tensor<128256x4096xf32>
    %3 = "ttir.mesh_shard"(%arg1, %2) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128256x4096xf32>, tensor<128256x4096xf32>) -> tensor<128256x4096xf32>
    %4 = ttir.empty() : tensor<14xi64>
    %5 = "ttir.mesh_shard"(%arg2, %4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<14xi64>, tensor<14xi64>) -> tensor<14xi64>
    %6 = ttir.empty() : tensor<64xf32>
    %7 = "ttir.mesh_shard"(%arg3, %6) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %8 = ttir.empty() : tensor<128x4096xf32>
    %9 = "ttir.mesh_shard"(%arg4, %8) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1024x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xf32>
    %10 = ttir.empty() : tensor<f32>
    %11 = "ttir.mesh_shard"(%arg5, %10) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %12 = ttir.empty() : tensor<4096xf32>
    %13 = "ttir.mesh_shard"(%arg6, %12) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>
    %14 = ttir.empty() : tensor<128x4096xf32>
    %15 = "ttir.mesh_shard"(%arg7, %14) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1024x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xf32>
    %16 = ttir.empty() : tensor<4096x1792xf32>
    %17 = "ttir.mesh_shard"(%arg8, %16) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 8>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096x14336xf32>, tensor<4096x1792xf32>) -> tensor<4096x1792xf32>
    %18 = ttir.empty() : tensor<1792x4096xf32>
    %19 = "ttir.mesh_shard"(%arg9, %18) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<14336x4096xf32>, tensor<1792x4096xf32>) -> tensor<1792x4096xf32>
    %20 = ttir.empty() : tensor<4096x512xf32>
    %21 = "ttir.mesh_shard"(%arg10, %20) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 8>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096x4096xf32>, tensor<4096x512xf32>) -> tensor<4096x512xf32>
    %22 = ttir.empty() : tensor<f32>
    %23 = "ttir.mesh_shard"(%arg11, %22) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %24 = ttir.empty() : tensor<f32>
    %25 = "ttir.mesh_shard"(%arg12, %24) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %26 = ttir.empty() : tensor<512x4096xf32>
    %27 = "ttir.mesh_shard"(%arg13, %26) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096x4096xf32>, tensor<512x4096xf32>) -> tensor<512x4096xf32>
    %28 = ttir.empty() : tensor<4096xf32>
    %29 = "ttir.mesh_shard"(%arg14, %28) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>
    %30 = ttir.empty() : tensor<1792x4096xf32>
    %31 = "ttir.mesh_shard"(%arg15, %30) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<14336x4096xf32>, tensor<1792x4096xf32>) -> tensor<1792x4096xf32>
    %32 = ttir.empty() : tensor<4096xf32>
    %33 = "ttir.mesh_shard"(%arg16, %32) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>
    %34 = ttir.empty() : tensor<16032x4096xf32>
    %35 = "ttir.mesh_shard"(%arg17, %34) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128256x4096xf32>, tensor<16032x4096xf32>) -> tensor<16032x4096xf32>
    %36 = "ttir.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]> : tensor<19xi64>}> : () -> tensor<19xi64>
    %37 = "ttir.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %38 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %39 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %40 = "ttir.constant"() <{value = dense<2.44140625E-4> : tensor<1x14xf32>}> : () -> tensor<1x14xf32>
    %41 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %42 = ttir.empty() : tensor<1x1xf32>
    %43 = "ttir.reshape"(%41, %42) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %44 = ttir.empty() : tensor<14x19xf32>
    %45 = "ttir.broadcast"(%43, %44) <{broadcast_dimensions = array<i64: 14, 19>}> : (tensor<1x1xf32>, tensor<14x19xf32>) -> tensor<14x19xf32>
    %46 = ttir.empty() : tensor<1x1x1xf32>
    %47 = "ttir.reshape"(%39, %46) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %48 = ttir.empty() : tensor<1x14x4096xf32>
    %49 = "ttir.broadcast"(%47, %48) <{broadcast_dimensions = array<i64: 1, 14, 4096>}> : (tensor<1x1x1xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %50 = ttir.empty() : tensor<1x1x1x1xbf16>
    %51 = "ttir.reshape"(%38, %50) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1x1x1xbf16>) -> tensor<1x1x1x1xbf16>
    %52 = ttir.empty() : tensor<1x1x19x128xbf16>
    %53 = "ttir.broadcast"(%51, %52) <{broadcast_dimensions = array<i64: 1, 1, 19, 128>}> : (tensor<1x1x1x1xbf16>, tensor<1x1x19x128xbf16>) -> tensor<1x1x19x128xbf16>
    %54 = ttir.empty() : tensor<1x14xui32>
    %55 = "ttir.typecast"(%1, %54) <{conservative_folding = false}> : (tensor<1x14xi64>, tensor<1x14xui32>) -> tensor<1x14xui32>
    %56 = ttir.empty() : tensor<14xui32>
    %57 = "ttir.reshape"(%55, %56) <{shape = [14 : i32]}> : (tensor<1x14xui32>, tensor<14xui32>) -> tensor<14xui32>
    %58 = ttir.empty() : tensor<14x4096xf32>
    %59 = "ttir.gather"(%3, %57, %58) <{collapsed_slice_dims = array<i64: 0>, index_vector_dim = 1 : si64, indices_are_sorted = false, offset_dims = array<i64: 1>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 4096>, start_index_map = array<i64: 0>, start_indices_batching_dims = array<i64>}> : (tensor<128256x4096xf32>, tensor<14xui32>, tensor<14x4096xf32>) -> tensor<14x4096xf32>
    %60 = ttir.empty() : tensor<1x14x4096xf32>
    %61 = "ttir.reshape"(%59, %60) <{shape = [1 : i32, 14 : i32, 4096 : i32]}> : (tensor<14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %62 = ttir.empty() : tensor<1x1x4096xf32>
    %63 = "ttir.reshape"(%13, %62) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<4096xf32>, tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %64 = ttir.empty() : tensor<1x14x4096xf32>
    %65 = "ttir.broadcast"(%63, %64) <{broadcast_dimensions = array<i64: 1, 14, 1>}> : (tensor<1x1x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %66 = ttir.empty() : tensor<1x14x4096xf32>
    %67 = "ttir.pow"(%61, %49, %66) : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %68 = ttir.empty() : tensor<1x14xf32>
    %69 = "ttir.sum"(%67, %68) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x14x4096xf32>, tensor<1x14xf32>) -> tensor<1x14xf32>
    %70 = ttir.empty() : tensor<1x14xf32>
    %71 = "ttir.multiply"(%69, %40, %70) : (tensor<1x14xf32>, tensor<1x14xf32>, tensor<1x14xf32>) -> tensor<1x14xf32>
    %72 = ttir.empty() : tensor<1x14x1xf32>
    %73 = "ttir.reshape"(%71, %72) <{shape = [1 : i32, 14 : i32, 1 : i32]}> : (tensor<1x14xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %74 = ttir.empty() : tensor<1x1x1xf32>
    %75 = "ttir.reshape"(%11, %74) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %76 = ttir.empty() : tensor<1x14x1xf32>
    %77 = "ttir.broadcast"(%75, %76) <{broadcast_dimensions = array<i64: 1, 14, 1>}> : (tensor<1x1x1xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %78 = ttir.empty() : tensor<1x14x1xf32>
    %79 = "ttir.add"(%73, %77, %78) : (tensor<1x14x1xf32>, tensor<1x14x1xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %80 = ttir.empty() : tensor<1x14x1xf32>
    %81 = "ttir.rsqrt"(%79, %80) : (tensor<1x14x1xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %82 = ttir.empty() : tensor<1x14xf32>
    %83 = "ttir.reshape"(%81, %82) <{shape = [1 : i32, 14 : i32]}> : (tensor<1x14x1xf32>, tensor<1x14xf32>) -> tensor<1x14xf32>
    %84 = ttir.empty() : tensor<1x14x1xf32>
    %85 = "ttir.reshape"(%83, %84) <{shape = [1 : i32, 14 : i32, 1 : i32]}> : (tensor<1x14xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %86 = ttir.empty() : tensor<1x14x4096xf32>
    %87 = "ttir.broadcast"(%85, %86) <{broadcast_dimensions = array<i64: 1, 1, 4096>}> : (tensor<1x14x1xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %88 = ttir.empty() : tensor<1x14x4096xf32>
    %89 = "ttir.multiply"(%61, %87, %88) : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %90 = ttir.empty() : tensor<1x14x4096xf32>
    %91 = "ttir.multiply"(%65, %89, %90) : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %92 = ttir.empty() : tensor<14x4096xf32>
    %93 = "ttir.reshape"(%91, %92) <{shape = [14 : i32, 4096 : i32]}> : (tensor<1x14x4096xf32>, tensor<14x4096xf32>) -> tensor<14x4096xf32>
    %94 = ttir.empty() : tensor<4096x128xf32>
    %95 = "ttir.permute"(%9, %94) <{permutation = array<i64: 1, 0>}> : (tensor<128x4096xf32>, tensor<4096x128xf32>) -> tensor<4096x128xf32>
    %96 = "ttir.dot_general"(%93, %95) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<14x4096xf32>, tensor<4096x128xf32>) -> tensor<14x128xf32>
    %97 = ttir.empty() : tensor<1x14x1x128xf32>
    %98 = "ttir.reshape"(%96, %97) <{shape = [1 : i32, 14 : i32, 1 : i32, 128 : i32]}> : (tensor<14x128xf32>, tensor<1x14x1x128xf32>) -> tensor<1x14x1x128xf32>
    %99 = ttir.empty() : tensor<1x1x14x128xf32>
    %100 = "ttir.permute"(%98, %99) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x14x1x128xf32>, tensor<1x1x14x128xf32>) -> tensor<1x1x14x128xf32>
    %101 = ttir.empty() : tensor<1x64x1xf32>
    %102 = "ttir.reshape"(%7, %101) <{shape = [1 : i32, 64 : i32, 1 : i32]}> : (tensor<64xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %103 = ttir.empty() : tensor<14xf32>
    %104 = "ttir.typecast"(%5, %103) <{conservative_folding = false}> : (tensor<14xi64>, tensor<14xf32>) -> tensor<14xf32>
    %105 = ttir.empty() : tensor<1x1x14xf32>
    %106 = "ttir.reshape"(%104, %105) <{shape = [1 : i32, 1 : i32, 14 : i32]}> : (tensor<14xf32>, tensor<1x1x14xf32>) -> tensor<1x1x14xf32>
    %107 = "ttir.dot_general"(%102, %106) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<1x64x1xf32>, tensor<1x1x14xf32>) -> tensor<1x64x14xf32>
    %108 = ttir.empty() : tensor<1x14x64xf32>
    %109 = "ttir.permute"(%107, %108) <{permutation = array<i64: 0, 2, 1>}> : (tensor<1x64x14xf32>, tensor<1x14x64xf32>) -> tensor<1x14x64xf32>
    %110 = ttir.empty() : tensor<1x14x128xf32>
    %111 = "ttir.concat"(%109, %109, %110) <{dim = 2 : si32}> : (tensor<1x14x64xf32>, tensor<1x14x64xf32>, tensor<1x14x128xf32>) -> tensor<1x14x128xf32>
    %112 = ttir.empty() : tensor<1x14x128xf32>
    %113 = "ttir.cos"(%111, %112) : (tensor<1x14x128xf32>, tensor<1x14x128xf32>) -> tensor<1x14x128xf32>
    %114 = ttir.empty() : tensor<1x1x14x128xf32>
    %115 = "ttir.reshape"(%113, %114) <{shape = [1 : i32, 1 : i32, 14 : i32, 128 : i32]}> : (tensor<1x14x128xf32>, tensor<1x1x14x128xf32>) -> tensor<1x1x14x128xf32>
    %116 = ttir.empty() : tensor<1x1x14x128xf32>
    %117 = "ttir.broadcast"(%115, %116) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x14x128xf32>, tensor<1x1x14x128xf32>) -> tensor<1x1x14x128xf32>
    %118 = ttir.empty() : tensor<1x1x14x128xf32>
    %119 = "ttir.multiply"(%100, %117, %118) : (tensor<1x1x14x128xf32>, tensor<1x1x14x128xf32>, tensor<1x1x14x128xf32>) -> tensor<1x1x14x128xf32>
    %120 = ttir.empty() : tensor<1x1x14x64xf32>
    %121 = "ttir.slice"(%100, %120) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 1 : i32, 14 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x14x128xf32>, tensor<1x1x14x64xf32>) -> tensor<1x1x14x64xf32>
    %122 = ttir.empty() : tensor<1x1x14x64xf32>
    %123 = "ttir.neg"(%121, %122) : (tensor<1x1x14x64xf32>, tensor<1x1x14x64xf32>) -> tensor<1x1x14x64xf32>
    %124 = ttir.empty() : tensor<1x1x14x64xf32>
    %125 = "ttir.slice"(%100, %124) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 14 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x14x128xf32>, tensor<1x1x14x64xf32>) -> tensor<1x1x14x64xf32>
    %126 = ttir.empty() : tensor<1x1x14x128xf32>
    %127 = "ttir.concat"(%123, %125, %126) <{dim = 3 : si32}> : (tensor<1x1x14x64xf32>, tensor<1x1x14x64xf32>, tensor<1x1x14x128xf32>) -> tensor<1x1x14x128xf32>
    %128 = ttir.empty() : tensor<1x14x128xf32>
    %129 = "ttir.sin"(%111, %128) : (tensor<1x14x128xf32>, tensor<1x14x128xf32>) -> tensor<1x14x128xf32>
    %130 = ttir.empty() : tensor<1x1x14x128xf32>
    %131 = "ttir.reshape"(%129, %130) <{shape = [1 : i32, 1 : i32, 14 : i32, 128 : i32]}> : (tensor<1x14x128xf32>, tensor<1x1x14x128xf32>) -> tensor<1x1x14x128xf32>
    %132 = ttir.empty() : tensor<1x1x14x128xf32>
    %133 = "ttir.broadcast"(%131, %132) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x14x128xf32>, tensor<1x1x14x128xf32>) -> tensor<1x1x14x128xf32>
    %134 = ttir.empty() : tensor<1x1x14x128xf32>
    %135 = "ttir.multiply"(%127, %133, %134) : (tensor<1x1x14x128xf32>, tensor<1x1x14x128xf32>, tensor<1x1x14x128xf32>) -> tensor<1x1x14x128xf32>
    %136 = ttir.empty() : tensor<1x1x14x128xf32>
    %137 = "ttir.add"(%119, %135, %136) : (tensor<1x1x14x128xf32>, tensor<1x1x14x128xf32>, tensor<1x1x14x128xf32>) -> tensor<1x1x14x128xf32>
    %138 = ttir.empty() : tensor<1x1x14x128xbf16>
    %139 = "ttir.typecast"(%137, %138) <{conservative_folding = false}> : (tensor<1x1x14x128xf32>, tensor<1x1x14x128xbf16>) -> tensor<1x1x14x128xbf16>
    %140 = ttir.empty() : tensor<1x1x19x128xbf16>
    %141 = "ttir.scatter"(%53, %5, %139, %140) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 2>, scatter_dims_to_operand_dims = array<i32: 2>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 0, 1, 3>}> : (tensor<1x1x19x128xbf16>, tensor<14xi64>, tensor<1x1x14x128xbf16>, tensor<1x1x19x128xbf16>) -> tensor<1x1x19x128xbf16>
    %142 = ttir.empty() : tensor<4096x128xf32>
    %143 = "ttir.permute"(%15, %142) <{permutation = array<i64: 1, 0>}> : (tensor<128x4096xf32>, tensor<4096x128xf32>) -> tensor<4096x128xf32>
    %144 = "ttir.dot_general"(%93, %143) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<14x4096xf32>, tensor<4096x128xf32>) -> tensor<14x128xf32>
    %145 = ttir.empty() : tensor<14x128xbf16>
    %146 = "ttir.typecast"(%144, %145) <{conservative_folding = false}> : (tensor<14x128xf32>, tensor<14x128xbf16>) -> tensor<14x128xbf16>
    %147 = ttir.empty() : tensor<1x14x1x128xbf16>
    %148 = "ttir.reshape"(%146, %147) <{shape = [1 : i32, 14 : i32, 1 : i32, 128 : i32]}> : (tensor<14x128xbf16>, tensor<1x14x1x128xbf16>) -> tensor<1x14x1x128xbf16>
    %149 = ttir.empty() : tensor<1x1x14x128xbf16>
    %150 = "ttir.permute"(%148, %149) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x14x1x128xbf16>, tensor<1x1x14x128xbf16>) -> tensor<1x1x14x128xbf16>
    %151 = ttir.empty() : tensor<1x1x19x128xbf16>
    %152 = "ttir.scatter"(%53, %5, %150, %151) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 2>, scatter_dims_to_operand_dims = array<i32: 2>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 0, 1, 3>}> : (tensor<1x1x19x128xbf16>, tensor<14xi64>, tensor<1x1x14x128xbf16>, tensor<1x1x19x128xbf16>) -> tensor<1x1x19x128xbf16>
    %153 = ttir.empty() : tensor<1x1x4096xf32>
    %154 = "ttir.reshape"(%33, %153) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<4096xf32>, tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %155 = ttir.empty() : tensor<1x14x4096xf32>
    %156 = "ttir.broadcast"(%154, %155) <{broadcast_dimensions = array<i64: 1, 14, 1>}> : (tensor<1x1x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %157 = ttir.empty() : tensor<4096x512xf32>
    %158 = "ttir.permute"(%27, %157) <{permutation = array<i64: 1, 0>}> : (tensor<512x4096xf32>, tensor<4096x512xf32>) -> tensor<4096x512xf32>
    %159 = "ttir.dot_general"(%93, %158) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<14x4096xf32>, tensor<4096x512xf32>) -> tensor<14x512xf32>
    %160 = ttir.empty() : tensor<1x14x4x128xf32>
    %161 = "ttir.reshape"(%159, %160) <{shape = [1 : i32, 14 : i32, 4 : i32, 128 : i32]}> : (tensor<14x512xf32>, tensor<1x14x4x128xf32>) -> tensor<1x14x4x128xf32>
    %162 = ttir.empty() : tensor<1x4x14x128xf32>
    %163 = "ttir.permute"(%161, %162) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x14x4x128xf32>, tensor<1x4x14x128xf32>) -> tensor<1x4x14x128xf32>
    %164 = ttir.empty() : tensor<1x1x14x128xf32>
    %165 = "ttir.reshape"(%113, %164) <{shape = [1 : i32, 1 : i32, 14 : i32, 128 : i32]}> : (tensor<1x14x128xf32>, tensor<1x1x14x128xf32>) -> tensor<1x1x14x128xf32>
    %166 = ttir.empty() : tensor<1x4x14x128xf32>
    %167 = "ttir.broadcast"(%165, %166) <{broadcast_dimensions = array<i64: 1, 4, 1, 1>}> : (tensor<1x1x14x128xf32>, tensor<1x4x14x128xf32>) -> tensor<1x4x14x128xf32>
    %168 = ttir.empty() : tensor<1x4x14x128xf32>
    %169 = "ttir.multiply"(%163, %167, %168) : (tensor<1x4x14x128xf32>, tensor<1x4x14x128xf32>, tensor<1x4x14x128xf32>) -> tensor<1x4x14x128xf32>
    %170 = ttir.empty() : tensor<1x4x14x64xf32>
    %171 = "ttir.slice"(%163, %170) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 4 : i32, 14 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4x14x128xf32>, tensor<1x4x14x64xf32>) -> tensor<1x4x14x64xf32>
    %172 = ttir.empty() : tensor<1x4x14x64xf32>
    %173 = "ttir.neg"(%171, %172) : (tensor<1x4x14x64xf32>, tensor<1x4x14x64xf32>) -> tensor<1x4x14x64xf32>
    %174 = ttir.empty() : tensor<1x4x14x64xf32>
    %175 = "ttir.slice"(%163, %174) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4 : i32, 14 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4x14x128xf32>, tensor<1x4x14x64xf32>) -> tensor<1x4x14x64xf32>
    %176 = ttir.empty() : tensor<1x4x14x128xf32>
    %177 = "ttir.concat"(%173, %175, %176) <{dim = 3 : si32}> : (tensor<1x4x14x64xf32>, tensor<1x4x14x64xf32>, tensor<1x4x14x128xf32>) -> tensor<1x4x14x128xf32>
    %178 = ttir.empty() : tensor<1x1x14x128xf32>
    %179 = "ttir.reshape"(%129, %178) <{shape = [1 : i32, 1 : i32, 14 : i32, 128 : i32]}> : (tensor<1x14x128xf32>, tensor<1x1x14x128xf32>) -> tensor<1x1x14x128xf32>
    %180 = ttir.empty() : tensor<1x4x14x128xf32>
    %181 = "ttir.broadcast"(%179, %180) <{broadcast_dimensions = array<i64: 1, 4, 1, 1>}> : (tensor<1x1x14x128xf32>, tensor<1x4x14x128xf32>) -> tensor<1x4x14x128xf32>
    %182 = ttir.empty() : tensor<1x4x14x128xf32>
    %183 = "ttir.multiply"(%177, %181, %182) : (tensor<1x4x14x128xf32>, tensor<1x4x14x128xf32>, tensor<1x4x14x128xf32>) -> tensor<1x4x14x128xf32>
    %184 = ttir.empty() : tensor<1x4x14x128xf32>
    %185 = "ttir.add"(%169, %183, %184) : (tensor<1x4x14x128xf32>, tensor<1x4x14x128xf32>, tensor<1x4x14x128xf32>) -> tensor<1x4x14x128xf32>
    %186 = ttir.empty() : tensor<4x14x128xf32>
    %187 = "ttir.reshape"(%185, %186) <{shape = [4 : i32, 14 : i32, 128 : i32]}> : (tensor<1x4x14x128xf32>, tensor<4x14x128xf32>) -> tensor<4x14x128xf32>
    %188 = ttir.empty() : tensor<1x1x1x19x128xbf16>
    %189 = "ttir.reshape"(%141, %188) <{shape = [1 : i32, 1 : i32, 1 : i32, 19 : i32, 128 : i32]}> : (tensor<1x1x19x128xbf16>, tensor<1x1x1x19x128xbf16>) -> tensor<1x1x1x19x128xbf16>
    %190 = ttir.empty() : tensor<1x1x4x19x128xbf16>
    %191 = "ttir.broadcast"(%189, %190) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x1x1x19x128xbf16>, tensor<1x1x4x19x128xbf16>) -> tensor<1x1x4x19x128xbf16>
    %192 = ttir.empty() : tensor<1x1x4x19x128xf32>
    %193 = "ttir.typecast"(%191, %192) <{conservative_folding = false}> : (tensor<1x1x4x19x128xbf16>, tensor<1x1x4x19x128xf32>) -> tensor<1x1x4x19x128xf32>
    %194 = ttir.empty() : tensor<1x4x19x128xf32>
    %195 = "ttir.reshape"(%193, %194) <{shape = [1 : i32, 4 : i32, 19 : i32, 128 : i32]}> : (tensor<1x1x4x19x128xf32>, tensor<1x4x19x128xf32>) -> tensor<1x4x19x128xf32>
    %196 = ttir.empty() : tensor<1x4x128x19xf32>
    %197 = "ttir.permute"(%195, %196) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x4x19x128xf32>, tensor<1x4x128x19xf32>) -> tensor<1x4x128x19xf32>
    %198 = ttir.empty() : tensor<4x128x19xf32>
    %199 = "ttir.reshape"(%197, %198) <{shape = [4 : i32, 128 : i32, 19 : i32]}> : (tensor<1x4x128x19xf32>, tensor<4x128x19xf32>) -> tensor<4x128x19xf32>
    %200 = "ttir.dot_general"(%187, %199) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<4x14x128xf32>, tensor<4x128x19xf32>) -> tensor<4x14x19xf32>
    %201 = ttir.empty() : tensor<1x4x14x19xf32>
    %202 = "ttir.reshape"(%200, %201) <{shape = [1 : i32, 4 : i32, 14 : i32, 19 : i32]}> : (tensor<4x14x19xf32>, tensor<1x4x14x19xf32>) -> tensor<1x4x14x19xf32>
    %203 = ttir.empty() : tensor<1x1x1x1xf32>
    %204 = "ttir.reshape"(%25, %203) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %205 = ttir.empty() : tensor<1x4x14x19xf32>
    %206 = "ttir.broadcast"(%204, %205) <{broadcast_dimensions = array<i64: 1, 4, 14, 19>}> : (tensor<1x1x1x1xf32>, tensor<1x4x14x19xf32>) -> tensor<1x4x14x19xf32>
    %207 = ttir.empty() : tensor<1x4x14x19xf32>
    %208 = "ttir.multiply"(%202, %206, %207) : (tensor<1x4x14x19xf32>, tensor<1x4x14x19xf32>, tensor<1x4x14x19xf32>) -> tensor<1x4x14x19xf32>
    %209 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 14 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<14xi32>
    %210 = ttir.empty() : tensor<14x1xi32>
    %211 = "ttir.reshape"(%209, %210) <{shape = [14 : i32, 1 : i32]}> : (tensor<14xi32>, tensor<14x1xi32>) -> tensor<14x1xi32>
    %212 = ttir.empty() : tensor<14x19xi32>
    %213 = "ttir.broadcast"(%211, %212) <{broadcast_dimensions = array<i64: 1, 19>}> : (tensor<14x1xi32>, tensor<14x19xi32>) -> tensor<14x19xi32>
    %214 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 19 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<19xi32>
    %215 = ttir.empty() : tensor<1x19xi32>
    %216 = "ttir.reshape"(%214, %215) <{shape = [1 : i32, 19 : i32]}> : (tensor<19xi32>, tensor<1x19xi32>) -> tensor<1x19xi32>
    %217 = ttir.empty() : tensor<14x19xi32>
    %218 = "ttir.broadcast"(%216, %217) <{broadcast_dimensions = array<i64: 14, 1>}> : (tensor<1x19xi32>, tensor<14x19xi32>) -> tensor<14x19xi32>
    %219 = ttir.empty() : tensor<14x19xi1>
    %220 = "ttir.ge"(%213, %218, %219) : (tensor<14x19xi32>, tensor<14x19xi32>, tensor<14x19xi1>) -> tensor<14x19xi1>
    %221 = ttir.empty() : tensor<1x1xf32>
    %222 = "ttir.reshape"(%23, %221) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %223 = ttir.empty() : tensor<14x19xf32>
    %224 = "ttir.broadcast"(%222, %223) <{broadcast_dimensions = array<i64: 14, 19>}> : (tensor<1x1xf32>, tensor<14x19xf32>) -> tensor<14x19xf32>
    %225 = ttir.empty() : tensor<14x19xf32>
    %226 = "ttir.where"(%220, %45, %224, %225) : (tensor<14x19xi1>, tensor<14x19xf32>, tensor<14x19xf32>, tensor<14x19xf32>) -> tensor<14x19xf32>
    %227 = ttir.empty() : tensor<1x19xi64>
    %228 = "ttir.reshape"(%36, %227) <{shape = [1 : i32, 19 : i32]}> : (tensor<19xi64>, tensor<1x19xi64>) -> tensor<1x19xi64>
    %229 = ttir.empty() : tensor<14x19xi64>
    %230 = "ttir.broadcast"(%228, %229) <{broadcast_dimensions = array<i64: 14, 1>}> : (tensor<1x19xi64>, tensor<14x19xi64>) -> tensor<14x19xi64>
    %231 = ttir.empty() : tensor<14x1xi64>
    %232 = "ttir.reshape"(%5, %231) <{shape = [14 : i32, 1 : i32]}> : (tensor<14xi64>, tensor<14x1xi64>) -> tensor<14x1xi64>
    %233 = ttir.empty() : tensor<14x19xi64>
    %234 = "ttir.broadcast"(%232, %233) <{broadcast_dimensions = array<i64: 1, 19>}> : (tensor<14x1xi64>, tensor<14x19xi64>) -> tensor<14x19xi64>
    %235 = ttir.empty() : tensor<14x19xi1>
    %236 = "ttir.gt"(%230, %234, %235) : (tensor<14x19xi64>, tensor<14x19xi64>, tensor<14x19xi1>) -> tensor<14x19xi1>
    %237 = ttir.empty() : tensor<14x19xf32>
    %238 = "ttir.typecast"(%236, %237) <{conservative_folding = false}> : (tensor<14x19xi1>, tensor<14x19xf32>) -> tensor<14x19xf32>
    %239 = ttir.empty() : tensor<14x19xf32>
    %240 = "ttir.multiply"(%226, %238, %239) : (tensor<14x19xf32>, tensor<14x19xf32>, tensor<14x19xf32>) -> tensor<14x19xf32>
    %241 = ttir.empty() : tensor<1x14x19xf32>
    %242 = "ttir.reshape"(%240, %241) <{shape = [1 : i32, 14 : i32, 19 : i32]}> : (tensor<14x19xf32>, tensor<1x14x19xf32>) -> tensor<1x14x19xf32>
    %243 = ttir.empty() : tensor<1x1x14x19xf32>
    %244 = "ttir.reshape"(%242, %243) <{shape = [1 : i32, 1 : i32, 14 : i32, 19 : i32]}> : (tensor<1x14x19xf32>, tensor<1x1x14x19xf32>) -> tensor<1x1x14x19xf32>
    %245 = ttir.empty() : tensor<1x4x14x19xf32>
    %246 = "ttir.broadcast"(%244, %245) <{broadcast_dimensions = array<i64: 1, 4, 1, 1>}> : (tensor<1x1x14x19xf32>, tensor<1x4x14x19xf32>) -> tensor<1x4x14x19xf32>
    %247 = ttir.empty() : tensor<1x4x14x19xf32>
    %248 = "ttir.add"(%208, %246, %247) : (tensor<1x4x14x19xf32>, tensor<1x4x14x19xf32>, tensor<1x4x14x19xf32>) -> tensor<1x4x14x19xf32>
    %249 = ttir.empty() : tensor<1x4x14xf32>
    %250 = "ttir.max"(%248, %249) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x4x14x19xf32>, tensor<1x4x14xf32>) -> tensor<1x4x14xf32>
    %251 = ttir.empty() : tensor<1x4x14x1xf32>
    %252 = "ttir.reshape"(%250, %251) <{shape = [1 : i32, 4 : i32, 14 : i32, 1 : i32]}> : (tensor<1x4x14xf32>, tensor<1x4x14x1xf32>) -> tensor<1x4x14x1xf32>
    %253 = ttir.empty() : tensor<1x4x14x19xf32>
    %254 = "ttir.broadcast"(%252, %253) <{broadcast_dimensions = array<i64: 1, 1, 1, 19>}> : (tensor<1x4x14x1xf32>, tensor<1x4x14x19xf32>) -> tensor<1x4x14x19xf32>
    %255 = ttir.empty() : tensor<1x4x14x19xf32>
    %256 = "ttir.subtract"(%248, %254, %255) : (tensor<1x4x14x19xf32>, tensor<1x4x14x19xf32>, tensor<1x4x14x19xf32>) -> tensor<1x4x14x19xf32>
    %257 = ttir.empty() : tensor<1x4x14x19xf32>
    %258 = "ttir.exp"(%256, %257) : (tensor<1x4x14x19xf32>, tensor<1x4x14x19xf32>) -> tensor<1x4x14x19xf32>
    %259 = ttir.empty() : tensor<1x4x14xf32>
    %260 = "ttir.sum"(%258, %259) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x4x14x19xf32>, tensor<1x4x14xf32>) -> tensor<1x4x14xf32>
    %261 = ttir.empty() : tensor<1x4x14x1xf32>
    %262 = "ttir.reshape"(%260, %261) <{shape = [1 : i32, 4 : i32, 14 : i32, 1 : i32]}> : (tensor<1x4x14xf32>, tensor<1x4x14x1xf32>) -> tensor<1x4x14x1xf32>
    %263 = ttir.empty() : tensor<1x4x14x19xf32>
    %264 = "ttir.broadcast"(%262, %263) <{broadcast_dimensions = array<i64: 1, 1, 1, 19>}> : (tensor<1x4x14x1xf32>, tensor<1x4x14x19xf32>) -> tensor<1x4x14x19xf32>
    %265 = ttir.empty() : tensor<1x4x14x19xf32>
    %266 = "ttir.div"(%258, %264, %265) : (tensor<1x4x14x19xf32>, tensor<1x4x14x19xf32>, tensor<1x4x14x19xf32>) -> tensor<1x4x14x19xf32>
    %267 = ttir.empty() : tensor<4x14x19xf32>
    %268 = "ttir.reshape"(%266, %267) <{shape = [4 : i32, 14 : i32, 19 : i32]}> : (tensor<1x4x14x19xf32>, tensor<4x14x19xf32>) -> tensor<4x14x19xf32>
    %269 = ttir.empty() : tensor<1x1x1x19x128xbf16>
    %270 = "ttir.reshape"(%152, %269) <{shape = [1 : i32, 1 : i32, 1 : i32, 19 : i32, 128 : i32]}> : (tensor<1x1x19x128xbf16>, tensor<1x1x1x19x128xbf16>) -> tensor<1x1x1x19x128xbf16>
    %271 = ttir.empty() : tensor<1x1x4x19x128xbf16>
    %272 = "ttir.broadcast"(%270, %271) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x1x1x19x128xbf16>, tensor<1x1x4x19x128xbf16>) -> tensor<1x1x4x19x128xbf16>
    %273 = ttir.empty() : tensor<1x1x4x19x128xf32>
    %274 = "ttir.typecast"(%272, %273) <{conservative_folding = false}> : (tensor<1x1x4x19x128xbf16>, tensor<1x1x4x19x128xf32>) -> tensor<1x1x4x19x128xf32>
    %275 = ttir.empty() : tensor<4x19x128xf32>
    %276 = "ttir.reshape"(%274, %275) <{shape = [4 : i32, 19 : i32, 128 : i32]}> : (tensor<1x1x4x19x128xf32>, tensor<4x19x128xf32>) -> tensor<4x19x128xf32>
    %277 = "ttir.dot_general"(%268, %276) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<4x14x19xf32>, tensor<4x19x128xf32>) -> tensor<4x14x128xf32>
    %278 = ttir.empty() : tensor<1x4x14x128xf32>
    %279 = "ttir.reshape"(%277, %278) <{shape = [1 : i32, 4 : i32, 14 : i32, 128 : i32]}> : (tensor<4x14x128xf32>, tensor<1x4x14x128xf32>) -> tensor<1x4x14x128xf32>
    %280 = ttir.empty() : tensor<1x14x4x128xf32>
    %281 = "ttir.permute"(%279, %280) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x4x14x128xf32>, tensor<1x14x4x128xf32>) -> tensor<1x14x4x128xf32>
    %282 = ttir.empty() : tensor<14x512xf32>
    %283 = "ttir.reshape"(%281, %282) <{shape = [14 : i32, 512 : i32]}> : (tensor<1x14x4x128xf32>, tensor<14x512xf32>) -> tensor<14x512xf32>
    %284 = ttir.empty() : tensor<512x4096xf32>
    %285 = "ttir.permute"(%21, %284) <{permutation = array<i64: 1, 0>}> : (tensor<4096x512xf32>, tensor<512x4096xf32>) -> tensor<512x4096xf32>
    %286 = "ttir.dot_general"(%283, %285) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<14x512xf32>, tensor<512x4096xf32>) -> tensor<14x4096xf32>
    %287 = ttir.empty() : tensor<14x4096xf32>
    %288 = "ttir.all_reduce"(%286, %287) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<14x4096xf32>, tensor<14x4096xf32>) -> tensor<14x4096xf32>
    %289 = ttir.empty() : tensor<1x14x4096xf32>
    %290 = "ttir.reshape"(%288, %289) <{shape = [1 : i32, 14 : i32, 4096 : i32]}> : (tensor<14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %291 = ttir.empty() : tensor<1x14x4096xf32>
    %292 = "ttir.add"(%61, %290, %291) : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %293 = ttir.empty() : tensor<1x1x4096xf32>
    %294 = "ttir.reshape"(%29, %293) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<4096xf32>, tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %295 = ttir.empty() : tensor<1x14x4096xf32>
    %296 = "ttir.broadcast"(%294, %295) <{broadcast_dimensions = array<i64: 1, 14, 1>}> : (tensor<1x1x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %297 = ttir.empty() : tensor<1x14x4096xf32>
    %298 = "ttir.pow"(%292, %49, %297) : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %299 = ttir.empty() : tensor<1x14xf32>
    %300 = "ttir.sum"(%298, %299) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x14x4096xf32>, tensor<1x14xf32>) -> tensor<1x14xf32>
    %301 = ttir.empty() : tensor<1x14xf32>
    %302 = "ttir.multiply"(%300, %40, %301) : (tensor<1x14xf32>, tensor<1x14xf32>, tensor<1x14xf32>) -> tensor<1x14xf32>
    %303 = ttir.empty() : tensor<1x14x1xf32>
    %304 = "ttir.reshape"(%302, %303) <{shape = [1 : i32, 14 : i32, 1 : i32]}> : (tensor<1x14xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %305 = ttir.empty() : tensor<1x14x1xf32>
    %306 = "ttir.add"(%304, %77, %305) : (tensor<1x14x1xf32>, tensor<1x14x1xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %307 = ttir.empty() : tensor<1x14x1xf32>
    %308 = "ttir.rsqrt"(%306, %307) : (tensor<1x14x1xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %309 = ttir.empty() : tensor<1x14xf32>
    %310 = "ttir.reshape"(%308, %309) <{shape = [1 : i32, 14 : i32]}> : (tensor<1x14x1xf32>, tensor<1x14xf32>) -> tensor<1x14xf32>
    %311 = ttir.empty() : tensor<1x14x1xf32>
    %312 = "ttir.reshape"(%310, %311) <{shape = [1 : i32, 14 : i32, 1 : i32]}> : (tensor<1x14xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %313 = ttir.empty() : tensor<1x14x4096xf32>
    %314 = "ttir.broadcast"(%312, %313) <{broadcast_dimensions = array<i64: 1, 1, 4096>}> : (tensor<1x14x1xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %315 = ttir.empty() : tensor<1x14x4096xf32>
    %316 = "ttir.multiply"(%292, %314, %315) : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %317 = ttir.empty() : tensor<1x14x4096xf32>
    %318 = "ttir.multiply"(%296, %316, %317) : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %319 = ttir.empty() : tensor<14x4096xf32>
    %320 = "ttir.reshape"(%318, %319) <{shape = [14 : i32, 4096 : i32]}> : (tensor<1x14x4096xf32>, tensor<14x4096xf32>) -> tensor<14x4096xf32>
    %321 = ttir.empty() : tensor<4096x1792xf32>
    %322 = "ttir.permute"(%31, %321) <{permutation = array<i64: 1, 0>}> : (tensor<1792x4096xf32>, tensor<4096x1792xf32>) -> tensor<4096x1792xf32>
    %323 = "ttir.dot_general"(%320, %322) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<14x4096xf32>, tensor<4096x1792xf32>) -> tensor<14x1792xf32>
    %324 = ttir.empty() : tensor<1x14x1792xf32>
    %325 = "ttir.reshape"(%323, %324) <{shape = [1 : i32, 14 : i32, 1792 : i32]}> : (tensor<14x1792xf32>, tensor<1x14x1792xf32>) -> tensor<1x14x1792xf32>
    %326 = ttir.empty() : tensor<1x14x1792xf32>
    %327 = "ttir.sigmoid"(%325, %326) : (tensor<1x14x1792xf32>, tensor<1x14x1792xf32>) -> tensor<1x14x1792xf32>
    %328 = ttir.empty() : tensor<1x14x1792xf32>
    %329 = "ttir.multiply"(%325, %327, %328) : (tensor<1x14x1792xf32>, tensor<1x14x1792xf32>, tensor<1x14x1792xf32>) -> tensor<1x14x1792xf32>
    %330 = ttir.empty() : tensor<4096x1792xf32>
    %331 = "ttir.permute"(%19, %330) <{permutation = array<i64: 1, 0>}> : (tensor<1792x4096xf32>, tensor<4096x1792xf32>) -> tensor<4096x1792xf32>
    %332 = "ttir.dot_general"(%320, %331) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<14x4096xf32>, tensor<4096x1792xf32>) -> tensor<14x1792xf32>
    %333 = ttir.empty() : tensor<1x14x1792xf32>
    %334 = "ttir.reshape"(%332, %333) <{shape = [1 : i32, 14 : i32, 1792 : i32]}> : (tensor<14x1792xf32>, tensor<1x14x1792xf32>) -> tensor<1x14x1792xf32>
    %335 = ttir.empty() : tensor<1x14x1792xf32>
    %336 = "ttir.multiply"(%329, %334, %335) : (tensor<1x14x1792xf32>, tensor<1x14x1792xf32>, tensor<1x14x1792xf32>) -> tensor<1x14x1792xf32>
    %337 = ttir.empty() : tensor<14x1792xf32>
    %338 = "ttir.reshape"(%336, %337) <{shape = [14 : i32, 1792 : i32]}> : (tensor<1x14x1792xf32>, tensor<14x1792xf32>) -> tensor<14x1792xf32>
    %339 = ttir.empty() : tensor<1792x4096xf32>
    %340 = "ttir.permute"(%17, %339) <{permutation = array<i64: 1, 0>}> : (tensor<4096x1792xf32>, tensor<1792x4096xf32>) -> tensor<1792x4096xf32>
    %341 = "ttir.dot_general"(%338, %340) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<14x1792xf32>, tensor<1792x4096xf32>) -> tensor<14x4096xf32>
    %342 = ttir.empty() : tensor<14x4096xf32>
    %343 = "ttir.all_reduce"(%341, %342) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<14x4096xf32>, tensor<14x4096xf32>) -> tensor<14x4096xf32>
    %344 = ttir.empty() : tensor<1x14x4096xf32>
    %345 = "ttir.reshape"(%343, %344) <{shape = [1 : i32, 14 : i32, 4096 : i32]}> : (tensor<14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %346 = ttir.empty() : tensor<1x14x4096xf32>
    %347 = "ttir.add"(%292, %345, %346) : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %348 = ttir.empty() : tensor<1x14x4096xf32>
    %349 = "ttir.pow"(%347, %49, %348) : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %350 = ttir.empty() : tensor<1x14xf32>
    %351 = "ttir.sum"(%349, %350) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x14x4096xf32>, tensor<1x14xf32>) -> tensor<1x14xf32>
    %352 = ttir.empty() : tensor<1x14xf32>
    %353 = "ttir.multiply"(%351, %40, %352) : (tensor<1x14xf32>, tensor<1x14xf32>, tensor<1x14xf32>) -> tensor<1x14xf32>
    %354 = ttir.empty() : tensor<1x14x1xf32>
    %355 = "ttir.reshape"(%353, %354) <{shape = [1 : i32, 14 : i32, 1 : i32]}> : (tensor<1x14xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %356 = ttir.empty() : tensor<1x14x1xf32>
    %357 = "ttir.add"(%355, %77, %356) : (tensor<1x14x1xf32>, tensor<1x14x1xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %358 = ttir.empty() : tensor<1x14x1xf32>
    %359 = "ttir.rsqrt"(%357, %358) : (tensor<1x14x1xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %360 = ttir.empty() : tensor<1x14xf32>
    %361 = "ttir.reshape"(%359, %360) <{shape = [1 : i32, 14 : i32]}> : (tensor<1x14x1xf32>, tensor<1x14xf32>) -> tensor<1x14xf32>
    %362 = ttir.empty() : tensor<1x14x1xf32>
    %363 = "ttir.reshape"(%361, %362) <{shape = [1 : i32, 14 : i32, 1 : i32]}> : (tensor<1x14xf32>, tensor<1x14x1xf32>) -> tensor<1x14x1xf32>
    %364 = ttir.empty() : tensor<1x14x4096xf32>
    %365 = "ttir.broadcast"(%363, %364) <{broadcast_dimensions = array<i64: 1, 1, 4096>}> : (tensor<1x14x1xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %366 = ttir.empty() : tensor<1x14x4096xf32>
    %367 = "ttir.multiply"(%347, %365, %366) : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %368 = ttir.empty() : tensor<1x14x4096xf32>
    %369 = "ttir.multiply"(%156, %367, %368) : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %370 = ttir.empty() : tensor<14x4096xf32>
    %371 = "ttir.reshape"(%369, %370) <{shape = [14 : i32, 4096 : i32]}> : (tensor<1x14x4096xf32>, tensor<14x4096xf32>) -> tensor<14x4096xf32>
    %372 = ttir.empty() : tensor<4096x16032xf32>
    %373 = "ttir.permute"(%35, %372) <{permutation = array<i64: 1, 0>}> : (tensor<16032x4096xf32>, tensor<4096x16032xf32>) -> tensor<4096x16032xf32>
    %374 = "ttir.dot_general"(%371, %373) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<14x4096xf32>, tensor<4096x16032xf32>) -> tensor<14x16032xf32>
    %375 = ttir.empty() : tensor<1x14x16032xf32>
    %376 = "ttir.reshape"(%374, %375) <{shape = [1 : i32, 14 : i32, 16032 : i32]}> : (tensor<14x16032xf32>, tensor<1x14x16032xf32>) -> tensor<1x14x16032xf32>
    %377 = ttir.empty() : tensor<1x14x4096xf32>
    %378 = "ttir.mesh_shard"(%61, %377) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %379 = ttir.empty() : tensor<1x8x19x128xbf16>
    %380 = "ttir.mesh_shard"(%141, %379) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 8, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x19x128xbf16>, tensor<1x8x19x128xbf16>) -> tensor<1x8x19x128xbf16>
    %381 = ttir.empty() : tensor<1x8x19x128xbf16>
    %382 = "ttir.mesh_shard"(%152, %381) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 8, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x19x128xbf16>, tensor<1x8x19x128xbf16>) -> tensor<1x8x19x128xbf16>
    %383 = ttir.empty() : tensor<1x14x4096xf32>
    %384 = "ttir.mesh_shard"(%369, %383) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<1x14x4096xf32>, tensor<1x14x4096xf32>) -> tensor<1x14x4096xf32>
    %385 = ttir.empty() : tensor<14x128256xf32>
    %386 = "ttir.mesh_shard"(%374, %385) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 8>, shard_type = #ttcore.shard_type<devices>}> : (tensor<14x16032xf32>, tensor<14x128256xf32>) -> tensor<14x128256xf32>
    %387 = ttir.empty() : tensor<1x14x128256xf32>
    %388 = "ttir.mesh_shard"(%376, %387) <{shard_dims = array<i64: -1, 2>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 8>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x14x16032xf32>, tensor<1x14x128256xf32>) -> tensor<1x14x128256xf32>
    return %378, %380, %382, %384, %386, %388 : tensor<1x14x4096xf32>, tensor<1x8x19x128xbf16>, tensor<1x8x19x128xbf16>, tensor<1x14x4096xf32>, tensor<14x128256xf32>, tensor<1x14x128256xf32>
  }
}
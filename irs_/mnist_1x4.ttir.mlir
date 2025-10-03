module @jit__lambda attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x4>]>} {
  func.func public @main(%arg0: tensor<1024xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<784x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<1024x512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg4: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg5: tensor<512x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg6: tensor<10xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg7: tensor<256x10xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg8: tensor<32x28x28x1xf32> {ttcore.argument_type = #ttcore.argument_type<input>}) -> (tensor<32x10xf32> {jax.result_info = "result"}) {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1024xbf16>) -> tensor<256xbf16>
    %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<784x1024xbf16>) -> tensor<784x256xbf16>
    %2 = "ttir.mesh_shard"(%arg2) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<512xbf16>) -> tensor<128xbf16>
    %3 = "ttir.mesh_shard"(%arg3) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1024x512xbf16>) -> tensor<1024x128xbf16>
    %4 = "ttir.mesh_shard"(%arg4) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<256xbf16>) -> tensor<64xbf16>
    %5 = "ttir.mesh_shard"(%arg5) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<512x256xbf16>) -> tensor<512x64xbf16>
    %6 = "ttir.mesh_shard"(%arg6) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<10xf32>) -> tensor<10xf32>
    %7 = "ttir.mesh_shard"(%arg7) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<256x10xf32>) -> tensor<256x10xf32>
    %8 = "ttir.mesh_shard"(%arg8) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<32x28x28x1xf32>) -> tensor<32x28x28x1xf32>
    %9 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %10 = "ttir.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %11 = ttir.empty() : tensor<32x784xf32>
    %12 = "ttir.reshape"(%8, %11) <{shape = [32 : i32, 784 : i32]}> : (tensor<32x28x28x1xf32>, tensor<32x784xf32>) -> tensor<32x784xf32>
    %13 = ttir.empty() : tensor<784x256xf32>
    %14 = "ttir.typecast"(%1, %13) <{conservative_folding = false}> : (tensor<784x256xbf16>, tensor<784x256xf32>) -> tensor<784x256xf32>
    %15 = ttir.empty() : tensor<256xf32>
    %16 = "ttir.typecast"(%0, %15) <{conservative_folding = false}> : (tensor<256xbf16>, tensor<256xf32>) -> tensor<256xf32>
    %17 = "ttir.dot_general"(%12, %14) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x784xf32>, tensor<784x256xf32>) -> tensor<32x256xf32>
    %18 = ttir.empty() : tensor<1x256xf32>
    %19 = "ttir.reshape"(%16, %18) <{shape = [1 : i32, 256 : i32]}> : (tensor<256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
    %20 = ttir.empty() : tensor<32x256xf32>
    %21 = "ttir.broadcast"(%19, %20) <{broadcast_dimensions = array<i64: 32, 1>}> : (tensor<1x256xf32>, tensor<32x256xf32>) -> tensor<32x256xf32>
    %22 = ttir.empty() : tensor<32x256xf32>
    %23 = "ttir.add"(%17, %21, %22) : (tensor<32x256xf32>, tensor<32x256xf32>, tensor<32x256xf32>) -> tensor<32x256xf32>
    %24 = ttir.empty() : tensor<1x1xf32>
    %25 = "ttir.reshape"(%9, %24) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %26 = ttir.empty() : tensor<32x256xf32>
    %27 = "ttir.broadcast"(%25, %26) <{broadcast_dimensions = array<i64: 32, 256>}> : (tensor<1x1xf32>, tensor<32x256xf32>) -> tensor<32x256xf32>
    %28 = ttir.empty() : tensor<32x256xf32>
    %29 = "ttir.maximum"(%23, %27, %28) : (tensor<32x256xf32>, tensor<32x256xf32>, tensor<32x256xf32>) -> tensor<32x256xf32>
    %30 = ttir.empty() : tensor<32x1024xf32>
    %31 = "ttir.all_gather"(%29, %30) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x256xf32>, tensor<32x1024xf32>) -> tensor<32x1024xf32>
    %32 = ttir.empty() : tensor<1024x128xf32>
    %33 = "ttir.typecast"(%3, %32) <{conservative_folding = false}> : (tensor<1024x128xbf16>, tensor<1024x128xf32>) -> tensor<1024x128xf32>
    %34 = ttir.empty() : tensor<128xf32>
    %35 = "ttir.typecast"(%2, %34) <{conservative_folding = false}> : (tensor<128xbf16>, tensor<128xf32>) -> tensor<128xf32>
    %36 = "ttir.dot_general"(%31, %33) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x1024xf32>, tensor<1024x128xf32>) -> tensor<32x128xf32>
    %37 = ttir.empty() : tensor<1x128xf32>
    %38 = "ttir.reshape"(%35, %37) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %39 = ttir.empty() : tensor<32x128xf32>
    %40 = "ttir.broadcast"(%38, %39) <{broadcast_dimensions = array<i64: 32, 1>}> : (tensor<1x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    %41 = ttir.empty() : tensor<32x128xf32>
    %42 = "ttir.add"(%36, %40, %41) : (tensor<32x128xf32>, tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    %43 = ttir.empty() : tensor<1x1xf32>
    %44 = "ttir.reshape"(%9, %43) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %45 = ttir.empty() : tensor<32x128xf32>
    %46 = "ttir.broadcast"(%44, %45) <{broadcast_dimensions = array<i64: 32, 128>}> : (tensor<1x1xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    %47 = ttir.empty() : tensor<32x128xf32>
    %48 = "ttir.maximum"(%42, %46, %47) : (tensor<32x128xf32>, tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    %49 = ttir.empty() : tensor<32x512xf32>
    %50 = "ttir.all_gather"(%48, %49) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xf32>, tensor<32x512xf32>) -> tensor<32x512xf32>
    %51 = ttir.empty() : tensor<512x64xf32>
    %52 = "ttir.typecast"(%5, %51) <{conservative_folding = false}> : (tensor<512x64xbf16>, tensor<512x64xf32>) -> tensor<512x64xf32>
    %53 = ttir.empty() : tensor<64xf32>
    %54 = "ttir.typecast"(%4, %53) <{conservative_folding = false}> : (tensor<64xbf16>, tensor<64xf32>) -> tensor<64xf32>
    %55 = "ttir.dot_general"(%50, %52) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x512xf32>, tensor<512x64xf32>) -> tensor<32x64xf32>
    %56 = ttir.empty() : tensor<1x64xf32>
    %57 = "ttir.reshape"(%54, %56) <{shape = [1 : i32, 64 : i32]}> : (tensor<64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
    %58 = ttir.empty() : tensor<32x64xf32>
    %59 = "ttir.broadcast"(%57, %58) <{broadcast_dimensions = array<i64: 32, 1>}> : (tensor<1x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %60 = ttir.empty() : tensor<32x64xf32>
    %61 = "ttir.add"(%55, %59, %60) : (tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %62 = ttir.empty() : tensor<1x1xf32>
    %63 = "ttir.reshape"(%9, %62) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %64 = ttir.empty() : tensor<32x64xf32>
    %65 = "ttir.broadcast"(%63, %64) <{broadcast_dimensions = array<i64: 32, 64>}> : (tensor<1x1xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %66 = ttir.empty() : tensor<32x64xf32>
    %67 = "ttir.maximum"(%61, %65, %66) : (tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %68 = ttir.empty() : tensor<32x256xf32>
    %69 = "ttir.all_gather"(%67, %68) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x64xf32>, tensor<32x256xf32>) -> tensor<32x256xf32>
    %70 = "ttir.dot_general"(%69, %7) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x256xf32>, tensor<256x10xf32>) -> tensor<32x10xf32>
    %71 = ttir.empty() : tensor<1x10xf32>
    %72 = "ttir.reshape"(%6, %71) <{shape = [1 : i32, 10 : i32]}> : (tensor<10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %73 = ttir.empty() : tensor<32x10xf32>
    %74 = "ttir.broadcast"(%72, %73) <{broadcast_dimensions = array<i64: 32, 1>}> : (tensor<1x10xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %75 = ttir.empty() : tensor<32x10xf32>
    %76 = "ttir.add"(%70, %74, %75) : (tensor<32x10xf32>, tensor<32x10xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %77 = ttir.empty() : tensor<32xf32>
    %78 = "ttir.max"(%76, %77) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x10xf32>, tensor<32xf32>) -> tensor<32xf32>
    %79 = ttir.empty() : tensor<1xf32>
    %80 = "ttir.reshape"(%10, %79) <{shape = [1 : i32]}> : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
    %81 = ttir.empty() : tensor<32xf32>
    %82 = "ttir.broadcast"(%80, %81) <{broadcast_dimensions = array<i64: 32>}> : (tensor<1xf32>, tensor<32xf32>) -> tensor<32xf32>
    %83 = ttir.empty() : tensor<32xf32>
    %84 = "ttir.maximum"(%82, %78, %83) : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %85 = ttir.empty() : tensor<32x1xf32>
    %86 = "ttir.reshape"(%84, %85) <{shape = [32 : i32, 1 : i32]}> : (tensor<32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
    %87 = ttir.empty() : tensor<32x1xf32>
    %88 = "ttir.broadcast"(%86, %87) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
    %89 = ttir.empty() : tensor<32x10xf32>
    %90 = "ttir.broadcast"(%88, %89) <{broadcast_dimensions = array<i64: 1, 10>}> : (tensor<32x1xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %91 = ttir.empty() : tensor<32x10xf32>
    %92 = "ttir.subtract"(%76, %90, %91) : (tensor<32x10xf32>, tensor<32x10xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %93 = ttir.empty() : tensor<32x10xf32>
    %94 = "ttir.exp"(%92, %93) : (tensor<32x10xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %95 = ttir.empty() : tensor<32xf32>
    %96 = "ttir.sum"(%94, %95) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x10xf32>, tensor<32xf32>) -> tensor<32xf32>
    %97 = ttir.empty() : tensor<32x1xf32>
    %98 = "ttir.reshape"(%96, %97) <{shape = [32 : i32, 1 : i32]}> : (tensor<32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
    %99 = ttir.empty() : tensor<32x1xf32>
    %100 = "ttir.broadcast"(%98, %99) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
    %101 = ttir.empty() : tensor<32x10xf32>
    %102 = "ttir.broadcast"(%100, %101) <{broadcast_dimensions = array<i64: 1, 10>}> : (tensor<32x1xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %103 = ttir.empty() : tensor<32x10xf32>
    %104 = "ttir.div"(%94, %102, %103) : (tensor<32x10xf32>, tensor<32x10xf32>, tensor<32x10xf32>) -> tensor<32x10xf32>
    %105 = "ttir.mesh_shard"(%104) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<32x10xf32>) -> tensor<32x10xf32>
    return %105 : tensor<32x10xf32>
  }
}


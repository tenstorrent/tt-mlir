// RUN: ttmlir-opt --canonicalize --ttir-implicit-broadcast-fold --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  module @gelu_erf attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
    func.func @main(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
      // CHECK: %[[GELU:.*]] = "ttir.gelu"(%arg0
      // CHECK: return %[[GELU]]
      %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
      %1 = "ttir.constant"() <{value = dense<7.070310e-01> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
      %2 = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
      %4 = "ttir.multiply"(%arg0, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %6 = "ttir.multiply"(%arg0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %8 = "ttir.erf"(%6) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %10 = "ttir.add"(%8, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %12 = "ttir.multiply"(%4, %10) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %12 : tensor<32x32xbf16>
    }
  }

  module @gelu_erf_arg_multiply attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
    func.func @main(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
      // CHECK: %[[MULTIPLIED_ARGS:.*]] = "ttir.multiply"(%arg0, %arg1
      // CHECK: %[[GELU:.*]] = "ttir.gelu"(%[[MULTIPLIED_ARGS]]
      // CHECK: return %[[GELU]]
      %multiplied_args = "ttir.multiply"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
      %1 = "ttir.constant"() <{value = dense<7.070310e-01> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
      %2 = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
      %4 = "ttir.multiply"(%multiplied_args, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %6 = "ttir.multiply"(%multiplied_args, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %8 = "ttir.erf"(%6) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %10 = "ttir.add"(%8, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %12 = "ttir.multiply"(%4, %10) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %12 : tensor<32x32xbf16>
    }
  }

  module @gelu_tanh attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
    func.func @main(%arg0: tensor<32x32xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<32x32xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
      // CHECK: %[[GELU:.*]] = "ttir.gelu"(%arg0
      // CHECK: return %[[GELU]]
      %0 = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
      %1 = "ttir.constant"() <{value = dense<4.467770e-02> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
      %2 = "ttir.constant"() <{value = dense<7.968750e-01> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
      %3 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
      %4 = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
      %6 = "ttir.multiply"(%arg0, %4) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %8 = "ttir.pow"(%arg0, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %10 = "ttir.multiply"(%8, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %12 = "ttir.add"(%arg0, %10) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %14 = "ttir.multiply"(%12, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %16 = "ttir.tanh"(%14) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %18 = "ttir.add"(%16, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %20 = "ttir.multiply"(%6, %18) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %20 : tensor<32x32xbf16>
    }
  }
}

module @gelu_tanh2 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  func.func public @main(%arg0: tensor<32x32xf32> {mhlo.sharding = "{replicated}", ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<32x32xf32> {jax.result_info = "result", ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    // CHECK: %[[GELU:.*]] = "ttir.gelu"(%arg0
    // CHECK: return %[[GELU]]
    %0 = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %1 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %2 = "ttir.constant"() <{value = dense<0.797884583> : tensor<f32>}> : () -> tensor<f32>
    %3 = "ttir.constant"() <{value = dense<4.471500e-02> : tensor<f32>}> : () -> tensor<f32>
    %5 = "ttir.multiply"(%arg0, %arg0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %7 = "ttir.multiply"(%5, %arg0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %9 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1xf32>
    %11 = "ttir.broadcast"(%9) <{broadcast_dimensions = array<i64: 32, 32>}> : (tensor<1x1xf32>) -> tensor<32x32xf32>
    %13 = "ttir.multiply"(%11, %7) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %15 = "ttir.add"(%arg0, %13) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %17 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1xf32>
    %19 = "ttir.broadcast"(%17) <{broadcast_dimensions = array<i64: 32, 32>}> : (tensor<1x1xf32>) -> tensor<32x32xf32>
    %21 = "ttir.multiply"(%19, %15) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %23 = "ttir.tanh"(%21) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    %25 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1xf32>
    %27 = "ttir.broadcast"(%25) <{broadcast_dimensions = array<i64: 32, 32>}> : (tensor<1x1xf32>) -> tensor<32x32xf32>
    %29 = "ttir.add"(%27, %23) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %31 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1xf32>
    %33 = "ttir.broadcast"(%31) <{broadcast_dimensions = array<i64: 32, 32>}> : (tensor<1x1xf32>) -> tensor<32x32xf32>
    %35 = "ttir.multiply"(%33, %29) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %37 = "ttir.multiply"(%arg0, %35) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %37 : tensor<32x32xf32>
  }
}

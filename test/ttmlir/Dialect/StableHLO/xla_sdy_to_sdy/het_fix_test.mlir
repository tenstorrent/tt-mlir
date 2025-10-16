// build/bin/ttmlir-opt -split-input-file --convert-xla-sdy-to-sdy test/ttmlir/Dialect/StableHLO/xla_sdy_to_sdy/het_fix_test.mlir

module @SyncTensorsGraph.420 attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2, \22_axis_1\22=4]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg13: tensor<39x39xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}) -> tensor<39x39xf32> {
    %6 = stablehlo.convert %arg13 {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"}, mhlo.sharding = "{replicated}"} : (tensor<39x39xbf16>) -> tensor<39x39xf32>
    return %6 : tensor<39x39xf32>
  }
}

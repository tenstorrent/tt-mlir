// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @erf_mhlo_op attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  func.func @main(%arg0: tensor<32x32xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<32x32xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    // CHECK: %[[ERF:.*]] = "ttir.erf"(%arg0
    // CHECK: return %[[ERF]]
    %0 = stablehlo.custom_call @mhlo.erf(%arg0) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}

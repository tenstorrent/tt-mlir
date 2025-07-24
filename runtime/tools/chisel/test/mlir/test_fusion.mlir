// Following MLIR file should showcase the MLIR op fusion
module @SyncTensorsGraph.8 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @transpose_matmul(%arg0: tensor<10x32x64xf32>, %arg1: tensor<10x32x64xf32>) -> tensor<10x64x64xf32> {
    %0 = ttir.empty() : tensor<10x64x32xf32>
    %1 = "ttir.permute"(%arg1, %0) <{permutation = array<i64: 0, 2, 1>}> : (tensor<10x32x64xf32>, tensor<10x64x32xf32>) -> tensor<10x64x32xf32>
    %2 = "ttir.dot_general"(%1, %arg0) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<10x64x32xf32>, tensor<10x32x64xf32>) -> tensor<10x64x64xf32>
    return %2 : tensor<10x64x64xf32>
  }
}

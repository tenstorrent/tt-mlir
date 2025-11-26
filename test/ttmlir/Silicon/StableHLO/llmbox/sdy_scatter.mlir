// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=1,8 automatic-arg-analysis" --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,8" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module @SyncTensorsGraph_1x8 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["_axis_0"=8]>
  func.func @main(%arg0: tensor<71x4xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}]>},
                   %arg1: tensor<71x4x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}, {?}]>},
                   %arg2: tensor<71x32xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"_axis_0", ?}]>}) -> tensor<71x32xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<71> : tensor<i64>
    %c_1 = stablehlo.constant dense<32> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<71x4xi64>
    %1 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<71x4xi64>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<71x4xi64>
    %3 = stablehlo.slice %arg1 [0:71, 0:4, 0:1] : (tensor<71x4x2xi64>) -> tensor<71x4x1xi64>
    %4 = stablehlo.reshape %3 : (tensor<71x4x1xi64>) -> tensor<71x4xi64>
    %5 = stablehlo.compare  LT, %4, %2 : (tensor<71x4xi64>, tensor<71x4xi64>) -> tensor<71x4xi1>
    %6 = stablehlo.add %4, %1 : tensor<71x4xi64>
    %7 = stablehlo.select %5, %6, %4 : tensor<71x4xi1>, tensor<71x4xi64>
    %8 = stablehlo.reshape %7 : (tensor<71x4xi64>) -> tensor<71x4x1xi64>
    %9 = stablehlo.slice %arg1 [0:71, 0:4, 1:2] : (tensor<71x4x2xi64>) -> tensor<71x4x1xi64>
    %10 = stablehlo.reshape %9 : (tensor<71x4x1xi64>) -> tensor<71x4xi64>
    %11 = stablehlo.compare  LT, %10, %2 : (tensor<71x4xi64>, tensor<71x4xi64>) -> tensor<71x4xi1>
    %12 = stablehlo.add %10, %0 : tensor<71x4xi64>
    %13 = stablehlo.select %11, %12, %10 : tensor<71x4xi1>, tensor<71x4xi64>
    %14 = stablehlo.reshape %13 : (tensor<71x4xi64>) -> tensor<71x4x1xi64>
    %15 = stablehlo.concatenate %8, %14, dim = 2 : (tensor<71x4x1xi64>, tensor<71x4x1xi64>) -> tensor<71x4x2xi64>
    %16 = "stablehlo.scatter"(%arg2, %15, %arg0) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>}> ({
    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
      stablehlo.return %arg4 : tensor<bf16>
    }) : (tensor<71x32xbf16>, tensor<71x4x2xi64>, tensor<71x4xbf16>) -> tensor<71x32xbf16>
    return %16 : tensor<71x32xbf16>
  }
}

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["batch"=1, "model"=2]>
    func.func @sharded_cache_update(%arg0: tensor<1x8x64x128xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22model\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "l__self___key_states"}, %arg1: tensor<64xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg2: tensor<i64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, []>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "auto_annotated_const_0"}, %arg3: tensor<1x8x1024x128xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22model\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}) -> tensor<1x8x1024x128xf32> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<64xi64>
    %1 = stablehlo.reshape %arg1 : (tensor<64xi64>) -> tensor<1x1x64xi64>
    %2 = stablehlo.reshape %1 : (tensor<1x1x64xi64>) -> tensor<64xi64>
    %3 = stablehlo.compare  LT, %2, %0 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i64>) -> tensor<64xi64>
    %5 = stablehlo.add %2, %4 : tensor<64xi64>
    %6 = stablehlo.select %3, %5, %2 : tensor<64xi1>, tensor<64xi64>
    %7 = stablehlo.reshape %6 : (tensor<64xi64>) -> tensor<64x1xi64>
    %8 = "stablehlo.scatter"(%arg3, %7, %arg0) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 3], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>}> ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
        stablehlo.return %arg5 : tensor<f32>
    }) : (tensor<1x8x1024x128xf32>, tensor<64x1xi64>, tensor<1x8x64x128xf32>) -> tensor<1x8x1024x128xf32>
    %9 = stablehlo.custom_call @Sharding(%8) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {\22model\22}, {}, {}]>]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}"} : (tensor<1x8x1024x128xf32>) -> tensor<1x8x1024x128xf32>
    return %9 : tensor<1x8x1024x128xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1, %arg2, %arg3) in_shardings=[<@mesh, [{}, {"model"}, {}, {}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, [{}, {"model"}, {}, {}]>] out_shardings=[<@mesh, [{}, {"model"}, {}, {}]>] manual_axes={"batch", "model"} (%arg4: tensor<1x4x64x128xf32>, %arg5: tensor<64xi64>, %arg6: tensor<i64>, %arg7: tensor<1x4x1024x128xf32>) {
// CHECK: "stablehlo.scatter"({{.*}}) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 3], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>}>
// CHECK: (tensor<1x[[DIM:4]]x1024x128xf32>, tensor<64x1xi64>, tensor<1x[[DIM]]x64x128xf32>) -> tensor<1x[[DIM]]x1024x128xf32>
// CHECK: sdy.return {{.*}} : tensor<1x[[DIM]]x1024x128xf32>
// CHECK: return {{.*}} : tensor<1x[[TWICEDIM:8]]x1024x128xf32>

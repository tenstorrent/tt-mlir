module @jit_scatter_reduction attributes {} {
    func.func public @test_scatter_with_add_reduction(%arg0: tensor<151936x2048xbf16>, %arg1: tensor<128x1xi64>, %arg2: tensor<128x2048xbf16>) -> tensor<151936x2048xbf16> {
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
        ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
            %add = stablehlo.maximum %arg3, %arg4 : tensor<bf16>
            stablehlo.return %add : tensor<bf16>
        }) : (tensor<151936x2048xbf16>, tensor<128x1xi64>, tensor<128x2048xbf16>) -> tensor<151936x2048xbf16>
        return %result : tensor<151936x2048xbf16>
    }
}

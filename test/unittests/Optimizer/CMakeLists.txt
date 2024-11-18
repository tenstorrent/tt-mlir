add_mlir_unittest(OptimizerTests
    TestShardSolver.cpp
)

target_link_libraries(OptimizerTests
    PRIVATE
    MLIR
    MLIRTTDialect
    MLIRTTNNPipelines
)

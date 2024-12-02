add_mlir_unittest(OptimizerTests
    TestShardSolver.cpp
    TestOptimizerOverrides.cpp
)

target_link_libraries(OptimizerTests
    PRIVATE
    MLIR
    MLIRTTDialect
    MLIRTTNNPipelines
)

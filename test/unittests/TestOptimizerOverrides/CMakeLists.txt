add_mlir_unittest(OptimizerOverridesTests
    TestOptimizerOverrides.cpp
)

target_link_libraries(OptimizerOverridesTests
    PRIVATE
    MLIR
    MLIRTTDialect
    MLIRTTIRDialect
    MLIRTTNNPipelines
)

add_mlir_unittest(OpModelTests
    TestOpModelLib.cpp
)

target_include_directories(OpModelTests
    PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
)

target_link_libraries(OpModelTests
    PRIVATE
    MLIR
    MLIRTTDialect
    MLIRTTNNAnalysis
    MLIRTTNNPipelines
    TTNNOpModelLib
)
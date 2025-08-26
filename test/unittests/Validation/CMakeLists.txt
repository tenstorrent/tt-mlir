add_mlir_unittest(ValidationTests
    TestOpConstraintValidator.cpp
    PARTIAL_SOURCES_INTENDED
)

target_link_libraries(ValidationTests
    PRIVATE
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTNNValidation
    TTMLIRTTNNUtils
    TTNNOpModelLib
    MLIRTTTransforms
)

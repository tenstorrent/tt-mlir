add_mlir_unittest(ValidationTests
    TestOpConstraintValidation.cpp
    PARTIAL_SOURCES_INTENDED
)

target_link_libraries(ValidationTests
    PRIVATE
    MLIRTTNNValidation
    MLIRTTTransforms
)

add_mlir_translation_library(TTNNTargetFlatbuffer
    TTNNToFlatbuffer.cpp
    TTNNToFlatbufferRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/TTNN

    LINK_LIBS PUBLIC
    MLIRTTNNDialect
    MLIRTTIRDialect
    MLIRTTDialect
    MLIRTTKernelDialect
    MLIRTTNNTransforms
    TTMLIRTTNNToEmitC
    MLIRQuantDialect
)

add_mlir_translation_library(TracedTTNNGraphToMLIR
    TracedTTNNGraphToMLIR.cpp
    TracedTTNNGraphToMLIRRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/TTNN

    LINK_LIBS PUBLIC
    MLIRTTNNDialect
    MLIRTTDialect
    MLIRTTNNTransforms
)

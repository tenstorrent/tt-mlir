add_mlir_translation_library(TTNNTargetFlatbuffer
    PARTIAL_SOURCES_INTENDED
    TTNNToFlatbuffer.cpp
    TTNNToFlatbufferRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/TTNN

    LINK_LIBS PUBLIC
    MLIRTTNNDialect
    MLIRTTIRDialect
    MLIRTTCoreDialect
    MLIRTTKernelDialect
    MLIRTTNNTransforms
    TTMLIRTTNNToEmitC
    MLIRQuantDialect
)

add_mlir_translation_library(TracedTTNNGraphToMLIR
    PARTIAL_SOURCES_INTENDED
    TracedTTNNGraphToMLIR.cpp
    TracedTTNNGraphToMLIRRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/TTNN

    LINK_LIBS PUBLIC
    MLIRTTNNDialect
    MLIRTTDialect
    MLIRTTNNTransforms
)

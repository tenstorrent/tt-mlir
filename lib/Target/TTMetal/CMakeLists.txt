add_mlir_translation_library(TTMetalTargetFlatbuffer
    TTMetalToFlatbuffer.cpp
    TTMetalToFlatbufferRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/TTMetal

    LINK_LIBS PUBLIC
    MLIRTTMetalDialect
    MLIRTTKernelDialect
    MLIRTTIRDialect
    MLIRTTCoreDialect
    TTMLIRTTKernelToEmitC
    MLIRD2MUtils
)

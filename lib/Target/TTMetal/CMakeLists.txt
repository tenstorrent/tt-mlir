add_mlir_translation_library(TTMetalTargetFlatbuffer
    TTMetalToFlatbuffer.cpp
    TTMetalToFlatbufferRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/TTMetal

    LINK_LIBS PUBLIC
    TTMLIRTTMetalDialect
    TTMLIRTTKernelDialect
    TTMLIRTTIRDialect
    TTMLIRTTDialect
    TTMLIRTTKernelToEmitC
)

target_include_directories(TTMetalTargetFlatbuffer PUBLIC ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)

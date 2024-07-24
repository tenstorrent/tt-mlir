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
)

target_include_directories(TTNNTargetFlatbuffer PUBLIC ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)

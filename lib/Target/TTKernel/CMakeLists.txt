add_mlir_translation_library(TTKernelTargetCpp
    TTKernelToCpp.cpp
    TTKernelToCppRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/TTKernel

    LINK_LIBS PUBLIC
    MLIRTTKernelDialect
    TTMLIRTTKernelToEmitC
    MLIRFuncToEmitC
    MLIRSCFToEmitC
    MLIRArithToEmitC
)

target_include_directories(TTKernelTargetCpp PUBLIC ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)

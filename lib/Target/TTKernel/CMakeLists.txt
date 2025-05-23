add_mlir_translation_library(TTKernelTargetCpp
    TTKernelToCpp.cpp
    TTKernelToCppRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/TTKernel

    DEPENDS
    TTKernelGeneratedLLKHeaders

    LINK_LIBS PUBLIC
    MLIRTTKernelDialect
    MLIRTTIRDialect
)

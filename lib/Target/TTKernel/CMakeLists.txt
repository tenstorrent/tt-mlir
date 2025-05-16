add_mlir_translation_library(TTKernelTargetCpp
    TTKernelToCpp.cpp
    TTKernelToCppRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/TTKernel

    LINK_LIBS PUBLIC
    MLIRTTKernelDialect
    MLIRTTIRDialect
)

add_dependencies(TTKernelTargetCpp TTKernelGeneratedLLKHeaders)

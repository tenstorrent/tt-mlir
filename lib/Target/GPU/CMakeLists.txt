add_mlir_translation_library(TTGPUTargetFlatbuffer
    GPUToFlatBuffer.cpp
    GPUToFlatbufferRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/GPU
    ${CMAKE_CURRENT_BINARY_DIR}/gpu_fbs

    DEPENDS
    GPU_FBS

    LINK_LIBS PUBLIC
    MLIRGPUDialect
    MLIRLLVMDialect
    MLIRNVVMDialect
    MLIRPass
    MLIRTransforms
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRFuncDialect

    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRNVVMToLLVMIRTranslation
    MLIRMemRefToLLVM
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRTranslateLib

    MLIRTTCoreDialect
    GPU_FBS

    LLVMSupport
    LLVMCore
)

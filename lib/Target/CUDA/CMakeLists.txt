add_mlir_translation_library(TTGPUTargetFlatbuffer
    CudaToFlatbuffer.cpp
    CudaToFlatbufferRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/CUDA
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_fbs

    DEPENDS

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
    MLIRReconcileUnrealizedCasts
    MLIRGPUToGPURuntimeTransforms
    TTMLIRTransforms
    MLIRNVVMToLLVM

    MLIRTTCoreDialect
    CUDA_FBS

    LLVMSupport
    LLVMCore
)

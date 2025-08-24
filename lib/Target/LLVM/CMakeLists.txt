add_mlir_translation_library(TTLLVMToDynamicLib
    LLVMToDynamicLib.cpp
    LLVMToDynamicLibRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${TTMLIR_SOURCE_DIR}/include/Target/TTMetal
    ${LLVM_INCLUDE_DIRS}

    LINK_LIBS PUBLIC
    MLIRLLVMDialect
    LLVMTarget
    LLVMX86Info
    LLVMX86AsmParser
    LLVMX86CodeGen
)

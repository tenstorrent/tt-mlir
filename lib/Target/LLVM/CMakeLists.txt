add_mlir_translation_library(TTLLVMToDynamicLib
    LLVMToDynamicLib.cpp
    LLVMToDynamicLibRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${TTMLIR_SOURCE_DIR}/include/Target/TTMetal

    LINK_LIBS PUBLIC
    LLVMCore
    LLVMSupport
    LLVMMC
    LLVMMCParser
    LLVMLinker
    LLVMBinaryFormat
    lldELF   # This is the key LLD library for ELF-based linking
    lldCommon
)

target_include_directories(TTMetalTargetFlatbuffer PUBLIC ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)

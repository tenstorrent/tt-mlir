set(LLVM_LIBS LLVMTarget)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR # Linux arm64
    CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR # macOS arm64
    CMAKE_OSX_ARCHITECTURES MATCHES "arm64")  # also macOS arm64
    list(APPEND LLVM_LIBS
        LLVMAArch64Info
        LLVMAArch64CodeGen
        LLVMAArch64AsmParser
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    list(APPEND LLVM_LIBS
        LLVMX86Info
        LLVMX86CodeGen
        LLVMX86AsmParser
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
    list(APPEND LLVM_LIBS
        LLVMPowerPCInfo
        LLVMPowerPCAsmParser
        LLVMPowerPCCodeGen
    )
else()
message(FATAL_ERROR "LLVM codegen/ASM parser libs: This HW architecture (${CMAKE_SYSTEM_PROCESSOR}) is not configured in cmake lib dependencies.")
endif()

add_mlir_translation_library(TTLLVMToDynamicLib
    LLVMToDynamicLib.cpp
    LLVMToDynamicLibRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${TTMLIR_SOURCE_DIR}/include/Target/TTMetal
    ${LLVM_INCLUDE_DIRS}

    LINK_LIBS PUBLIC
    MLIRLLVMDialect
    ${LLVM_LIBS}
)

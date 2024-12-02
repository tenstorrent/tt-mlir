set(_lld_deps)
# Use generator expressions to conditionally include LLD targets, providing
# a compiler definition if not available.
macro(_add_dependent_lld_target target not_definition)
list(APPEND _lld_deps
    $<TARGET_NAME_IF_EXISTS:${target}>
    $<IF:$<TARGET_EXISTS:${target}>,lldCommon,>
)
list(APPEND _lld_copts
    $<IF:$<TARGET_EXISTS:${target}>,,-D${not_definition}>)
endmacro()
_add_dependent_lld_target(lldCOFF IREE_COMPILER_LLD_COFF_DISABLED)
_add_dependent_lld_target(lldELF IREE_COMPILER_LLD_ELF_DISABLED)
_add_dependent_lld_target(lldMachO IREE_COMPILER_LLD_MACHO_DISABLED)
_add_dependent_lld_target(lldWasm IREE_COMPILER_LLD_WASM_DISABLED)

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
    MLIRBytecodeWriter
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    lldCommon
)

target_include_directories(TTMetalTargetFlatbuffer PUBLIC ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)

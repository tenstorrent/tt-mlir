add_mlir_dialect(TTOps tt)
add_mlir_doc(TTDialect TTDialect ttmlir/ -gen-dialect-doc)
add_mlir_doc(TTOps TTOps ttmlir/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TTPasses.td)
mlir_tablegen(TTPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTTPassesIncGen)

set(LLVM_TARGET_DEFINITIONS TTOpsEnums.td)
mlir_tablegen(TTOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TTOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTTOpsEnumsIncGen)
add_dependencies(mlir-headers MLIRTTOpsEnumsIncGen)

add_subdirectory(Target)
add_dependencies(mlir-headers TT_FBS)

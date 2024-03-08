add_mlir_dialect(TTOps ttmlir)
add_mlir_doc(TTDialect TTDialect ttmlir/ -gen-dialect-doc)
add_mlir_doc(TTOps TTOps ttmlir/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TTPasses.td)
mlir_tablegen(TTPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTTPassesIncGen)

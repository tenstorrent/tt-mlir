add_mlir_dialect(TTIROps ttir)
add_mlir_doc(TTIRDialect TTIRDialect ttmlir/ -gen-dialect-doc)
add_mlir_doc(TTIROps TTIROps ttmlir/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TTIRPasses.td)
mlir_tablegen(TTIRPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTTIRPassesIncGen)

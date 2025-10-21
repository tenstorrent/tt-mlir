set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRD2MPassesIncGen)
add_dependencies(mlir-headers MLIRD2MPassesIncGen)

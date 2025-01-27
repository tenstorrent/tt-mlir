set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRLLVMPassesIncGen)
add_dependencies(mlir-headers MLIRLLVMPassesIncGen)

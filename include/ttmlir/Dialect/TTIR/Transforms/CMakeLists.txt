set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTTIRPassesIncGen)
add_dependencies(mlir-headers MLIRTTIRPassesIncGen)
add_mlir_pdll_library(MLIRTTIRPatternsIncGen
  FuseRelu.pdll
  FuseRelu.h.inc
)
add_dependencies(mlir-headers MLIRTTIRPatternsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTTIRPassesIncGen)
add_dependencies(mlir-headers MLIRTTIRPassesIncGen)
add_mlir_pdll_library(MLIRTTIRPatternsIncGen
  FlattenZYReduce.pdll
  FlattenZYReducePatterns.h.inc
)

  add_dependencies(mlir-headers MLIRTTIRPatternsIncGen)

# Exclude tests from libMLIR.so
# add_mlir_library(MLIRTestPDLL2
#   TestPDLL2.cpp

#   EXCLUDE_FROM_LIBMLIR

#   ADDITIONAL_HEADER_DIRS
#   ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/PDLL

#   DEPENDS
#   MLIRTestPDLL2PatternsIncGen

#   LINK_LIBS PUBLIC
#   MLIRCastInterfaces
#   MLIRIR
#   MLIRPass
#   MLIRPDLInterpDialect
#   MLIRPDLDialect
#   MLIRSupport
#   MLIRTestDialect
#   MLIRTransformUtils
#   )

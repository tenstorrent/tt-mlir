add_mlir_dialect(TTNNOps ttnn)
add_mlir_doc(TTNNBase TTNNDialect src/autogen/md/Dialect/ -gen-dialect-doc)
add_mlir_doc(TTNNOps TTNNOp src/autogen/md/Dialect/ -gen-op-doc)

add_mlir_interface(TTNNOpModelInterface)
add_mlir_interface(TTNNWorkaroundInterface)

set(LLVM_TARGET_DEFINITIONS TTNNOpsEnums.td)
mlir_tablegen(TTNNOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TTNNOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTTNNOpsEnumsIncGen)
add_dependencies(mlir-headers MLIRTTNNOpsEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS TTNNOpsAttrs.td)
mlir_tablegen(TTNNOpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TTNNOpsAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTTNNOpsAttrsIncGen)
add_dependencies(mlir-headers MLIRTTNNOpsAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS TTNNOpsTypes.td)
mlir_tablegen(TTNNOpsTypeDefs.h.inc -gen-typedef-decls)
mlir_tablegen(TTNNOpsTypeDefs.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRTTNNOpsTypesIncGen)
add_dependencies(mlir-headers MLIRTTNNOpsTypesIncGen)

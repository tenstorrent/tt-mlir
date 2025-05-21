add_mlir_dialect(EmitPyOps emitpy)
add_mlir_doc(EmitPyBase EmitPyDialect src/autogen/md/Dialect/ -gen-dialect-doc)
add_mlir_doc(EmitPyOps EmitPyOp src/autogen/md/Dialect/ -gen-op-doc)

# set(LLVM_TARGET_DEFINITIONS EmitPyTypes.td)
# mlir_tablegen(EmitPyTypeDefs.h.inc -gen-typedef-decls)
# mlir_tablegen(EmitPyTypeDefs.cpp.inc -gen-typedef-defs)
# add_public_tablegen_target(MLIREmitPyTypesIncGen)
# add_dependencies(mlir-headers MLIREmitPyTypesIncGen)

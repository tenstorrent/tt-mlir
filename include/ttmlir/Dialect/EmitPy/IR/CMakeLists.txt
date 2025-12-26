# Add the Interfaces directory to tablegen include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Interfaces)

set(LLVM_TARGET_DEFINITIONS EmitPyOps.td)
add_mlir_dialect(EmitPyOps emitpy)
add_mlir_doc(EmitPyBase EmitPyDialect src/autogen/md/Dialect/ -gen-dialect-doc)
add_mlir_doc(EmitPyOps EmitPyOp src/autogen/md/Dialect/ -gen-op-doc)
add_dependencies(MLIREmitPyOpsIncGen MLIREmitPyInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS EmitPyAttrs.td)
mlir_tablegen(EmitPyAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(EmitPyAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIREmitPyAttrsIncGen)
add_dependencies(mlir-headers MLIREmitPyAttrsIncGen)

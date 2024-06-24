add_mlir_dialect(TTIROps ttir)
add_mlir_doc(TTIRBase TTIRDialect Dialect/ -gen-dialect-doc)
add_mlir_doc(TTIROps TTIROp Dialect/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TTIROpsInterfaces.td)
mlir_tablegen(TTIROpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TTIROpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(TTIROpsInterfacesIncGen)
add_dependencies(mlir-headers TTIROpsInterfacesIncGen)

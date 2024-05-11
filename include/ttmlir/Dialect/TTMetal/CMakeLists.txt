add_mlir_dialect(TTMetalOps ttmetal)
add_mlir_doc(TTMetalDialect TTMetalDialect ttmlir/ -gen-dialect-doc)
add_mlir_doc(TTMetalOps TTMetalOps ttmlir/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TTMetalOpsTypes.td)
mlir_tablegen(TTMetalOpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TTMetalOpsAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTTMetalOpsAttributesIncGen)
add_dependencies(mlir-headers MLIRTTMetalOpsAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS TTMetalOpsEnums.td)
mlir_tablegen(TTMetalOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TTMetalOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTTMetalOpsEnumsIncGen)
add_dependencies(mlir-headers MLIRTTMetalOpsEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS TTMetalPasses.td)
mlir_tablegen(TTMetalPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTTMetalPassesIncGen)

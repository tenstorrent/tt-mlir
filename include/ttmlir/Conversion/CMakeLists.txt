include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)
include_directories(${TTMLIR_SOURCE_DIR}/include)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TTMLIRConversion)
add_dependencies(mlir-headers PassesIncGen)

add_public_tablegen_target(TTMLIRConversionPassIncGen)
add_dependencies(mlir-headers TTMLIRConversionPassIncGen)

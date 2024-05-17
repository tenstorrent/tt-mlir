get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRTTDialect
        MLIRTTKernelDialect
        MLIRTTIRDialect
        MLIRTTMetalDialect
        MLIRTargetCpp
        )
add_llvm_executable(ttmlir-opt ttmlir-opt.cpp)

llvm_update_compile_flags(ttmlir-opt)
target_link_libraries(ttmlir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(ttmlir-opt)

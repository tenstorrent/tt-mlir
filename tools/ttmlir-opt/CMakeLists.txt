get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS ${dialect_libs} ${conversion_libs} ${extension_libs} MLIROptLib MLIRTargetCpp TTMLIRStatic)

add_llvm_executable(ttmlir-opt ttmlir-opt.cpp DISABLE_LLVM_LINK_LLVM_DYLIB)

llvm_update_compile_flags(ttmlir-opt)
target_link_libraries(ttmlir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(ttmlir-opt)

install(TARGETS ttmlir-opt DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)

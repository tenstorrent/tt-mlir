get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS ${dialect_libs} ${conversion_libs} ${extension_libs}
  MLIROptLib
  MLIRTargetCpp
  TTMLIRStatic
  MLIRLspServerLib
)

add_llvm_executable(ttmlir-lsp-server ttmlir-lsp-server.cpp DISABLE_LLVM_LINK_LLVM_DYLIB)
llvm_update_compile_flags(ttmlir-lsp-server)
target_link_libraries(ttmlir-lsp-server PRIVATE ${LIBS})

mlir_check_all_link_libraries(ttmlir-lsp-server)

install(TARGETS ttmlir-lsp-server DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)

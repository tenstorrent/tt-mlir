add_llvm_executable(ttmlir-lsp-server ttmlir-lsp-server.cpp)
llvm_update_compile_flags(ttmlir-lsp-server)
target_link_libraries(ttmlir-lsp-server PRIVATE MLIR TTMLIR)

mlir_check_all_link_libraries(ttmlir-lsp-server)

install(TARGETS ttmlir-lsp-server DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)

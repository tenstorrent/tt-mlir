add_llvm_executable(ttmlir-opt ttmlir-opt.cpp)

llvm_update_compile_flags(ttmlir-opt)
target_link_libraries(ttmlir-opt PRIVATE MLIR LLVM TTMLIR)

mlir_check_all_link_libraries(ttmlir-opt)

install(TARGETS ttmlir-opt DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)

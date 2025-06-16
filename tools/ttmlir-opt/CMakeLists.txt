add_llvm_executable(ttmlir-opt ttmlir-opt.cpp)

llvm_update_compile_flags(ttmlir-opt)
target_link_libraries(ttmlir-opt PRIVATE TTMLIRCompilerStatic MLIROptLib)
target_link_libraries(ttmlir-opt PUBLIC coverage_config)

if (TTMLIR_ENABLE_TTNN_PY)
    target_link_libraries(ttmlir-opt PRIVATE Python3::Python)
endif()

mlir_check_all_link_libraries(ttmlir-opt)

install(TARGETS ttmlir-opt DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)

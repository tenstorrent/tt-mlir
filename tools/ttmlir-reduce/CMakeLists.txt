add_llvm_executable(ttmlir-reduce ttmlir-reduce.cpp)
llvm_update_compile_flags(ttmlir-reduce)
target_link_libraries(ttmlir-reduce PRIVATE TTMLIRCompilerStatic MLIRReduceLib)
target_link_libraries(ttmlir-reduce PUBLIC coverage_config)

mlir_check_all_link_libraries(ttmlir-reduce)

install(TARGETS ttmlir-reduce DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)

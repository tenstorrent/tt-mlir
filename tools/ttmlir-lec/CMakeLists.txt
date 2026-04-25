add_llvm_executable(ttmlir-lec ttmlir-lec.cpp)

llvm_update_compile_flags(ttmlir-lec)
target_link_libraries(ttmlir-lec PRIVATE TTMLIRCompilerStatic)
target_link_libraries(ttmlir-lec PUBLIC coverage_config)

mlir_check_all_link_libraries(ttmlir-lec)

install(TARGETS ttmlir-lec DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)

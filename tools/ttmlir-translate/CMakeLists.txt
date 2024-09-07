add_llvm_executable(ttmlir-translate ttmlir-translate.cpp)

llvm_update_compile_flags(ttmlir-translate)
target_link_libraries(ttmlir-translate PRIVATE
  TTNNTargetFlatbuffer
  TTMetalTargetFlatbuffer
  MLIRTTNNTransforms
  TTMLIRTTNNToEmitC
  TTMLIRTTKernelToEmitC
)

mlir_check_link_libraries(ttmlir-translate)

install(TARGETS ttmlir-translate DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)

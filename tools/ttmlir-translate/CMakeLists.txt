get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(ttmlir-translate ttmlir-translate.cpp)

llvm_update_compile_flags(ttmlir-translate)
target_link_libraries(ttmlir-translate PRIVATE
  MLIR
  LLVM
  TTMLIRCompiler
)

mlir_check_link_libraries(ttmlir-translate)

install(TARGETS ttmlir-translate DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)

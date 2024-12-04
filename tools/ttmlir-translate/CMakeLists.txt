get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(target_libs GLOBAL PROPERTY LLVM_STATIC_LIBS)


add_llvm_executable(ttmlir-translate ttmlir-translate.cpp DISABLE_LLVM_LINK_LLVM_DYLIB)

llvm_update_compile_flags(ttmlir-translate)
target_link_libraries(ttmlir-translate PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${translation_libs}
  ${extension_libs}
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  ${target_libs}
)

mlir_check_link_libraries(ttmlir-translate)

install(TARGETS ttmlir-translate DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)

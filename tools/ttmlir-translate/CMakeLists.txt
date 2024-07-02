get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(ttmlir-translate ttmlir-translate.cpp)

llvm_update_compile_flags(ttmlir-translate)
target_link_libraries(ttmlir-translate
  PRIVATE
  ${translation_libs}
  )

mlir_check_link_libraries(ttmlir-translate)

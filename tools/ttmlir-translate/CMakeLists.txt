get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS ${translation_libs} ${dialect_libs} TTMLIRTTNNToEmitC TTNNTargetFlatbuffer)
add_llvm_executable(ttmlir-translate ttmlir-translate.cpp)

llvm_update_compile_flags(ttmlir-translate)
target_link_libraries(ttmlir-translate PRIVATE ${LIBS})

mlir_check_link_libraries(ttmlir-translate)

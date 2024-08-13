if (TTMLIR_ENABLE_STABLEHLO)
include(../../third_party/stablehlo_libs.cmake)
endif()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} ${stablehlo_libs} MLIROptLib MLIRTargetCpp TTMLIRStatic)

add_llvm_executable(ttmlir-opt ttmlir-opt.cpp)

if (TTMLIR_ENABLE_STABLEHLO)
add_dependencies(ttmlir-opt stablehlo)
endif()

llvm_update_compile_flags(ttmlir-opt)

target_link_libraries(ttmlir-opt PRIVATE ${LIBS})
target_link_directories(ttmlir-opt PUBLIC ${TTMLIR_TOOLCHAIN_DIR}/lib)
target_link_directories(ttmlir-opt PUBLIC ${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo-build/lib)

mlir_check_all_link_libraries(ttmlir-opt ${LIBS})

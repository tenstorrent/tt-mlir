get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} StablehloBase TTNNTargetFlatbuffer MLIRTTIRDialect TTMLIRConversions MLIRTransformUtils MLIROptLib MLIRTargetCpp TTMLIRStatic CheckOps ChloCAPI ChloOps InterpreterOps StablehloAssemblyFormat StablehloBase StablehloBroadcastUtils StablehloCAPI StablehloLinalgTransforms StablehloOps StablehloPasses StablehloPortableApi StablehloRegister StablehloSerialization StablehloTOSATransforms StablehloTestUtils StablehloTypeInference Version VhloCAPI VhloOps VhloTypes)

if (TTMLIR_ENABLE_STABLEHLO)
	list(APPEND LIBS TTMLIRStableHLOToTTIR)
endif()

add_llvm_executable(ttmlir-opt ttmlir-opt.cpp)

target_link_directories(ttmlir-opt PRIVATE ${TTMLIR_PROJECT_DIR}/build/lib)
target_link_libraries(ttmlir-opt PRIVATE ${LIBS} libStablehloBase.a)
target_link_directories(ttmlir-opt PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib)

llvm_update_compile_flags(ttmlir-opt)

mlir_check_all_link_libraries(ttmlir-opt ${LIBS})

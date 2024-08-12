
include(CMakePrintHelpers)

set(stablehlo_libs 
CheckOps
ChloCAPI
ChloOps
InterpreterOps
StablehloAssemblyFormat
StablehloBase
StablehloBroadcastUtils
StablehloCAPI
StablehloLinalgTransforms
StablehloOps
StablehloPasses
StablehloPortableApi
StablehloRegister
StablehloSerialization
StablehloTOSATransforms
StablehloTestUtils
StablehloTypeInference
Version
VhloCAPI
VhloOps
VhloTypes)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${stablehlo_libs} ${dialect_libs} ${conversion_libs} MLIROptLib MLIRTargetCpp TTMLIRStatic)

add_llvm_executable(ttmlir-opt ttmlir-opt.cpp)
add_dependencies(ttmlir-opt stablehlo)

llvm_update_compile_flags(ttmlir-opt)
target_link_directories(ttmlir-opt PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo-build/lib)

target_link_libraries(ttmlir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(ttmlir-opt ${LIBS})

set_property(GLOBAL PROPERTY STABLEHLO_LIBS "${STABLEHLO_LIBRARIES_LIST}")

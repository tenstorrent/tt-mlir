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
StablehloReferenceApi
StablehloReferenceAxes
StablehloReferenceConfiguration
StablehloReferenceElement
StablehloReferenceErrors
StablehloReferenceIndex
StablehloReferenceNumPy
StablehloReferenceOps
StablehloReferenceProcess
StablehloReferenceProcessGrid
StablehloReferenceScope
StablehloReferenceTensor
StablehloReferenceToken
StablehloReferenceTypes
StablehloReferenceValue
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
set(LIBS TTMLIRStableHLOToTTIR TTMLIRCAPI ${stablehlo_libs} ${dialect_libs} ${conversion_libs} MLIROptLib MLIRTargetCpp 
MLIRTTIRAnalysis
TTMLIRTTIRToTTNN
MLIRTTNNTransforms
TTMLIRTosaToTTIR
TTMLIRTTNNToEmitC
MLIRTTMetalTransforms
MLIRTTIRTransforms
MLIRTTMetalDialect
MLIRTTKernelDialect
MLIRTTIRDialect
MLIRTTNNDialect
MLIRTTNNPipelines
MLIRTTDialect
TTNNTargetFlatbuffer
TTMLIRCAPI
TTMLIRStableHLOToTTIR
TTMLIRStatic)

add_llvm_executable(ttmlir-opt ttmlir-opt.cpp)
add_dependencies(ttmlir-opt stablehlo)

llvm_update_compile_flags(ttmlir-opt)
target_link_directories(ttmlir-opt PUBLIC ${TTMLIR_TOOLCHAIN_DIR}/lib)
target_link_directories(ttmlir-opt PUBLIC ${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo-build/lib)

ADD_CUSTOM_COMMAND(TARGET ttmlir-opt
	PRE_BUILD
        COMMAND cp ${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo-build/lib/*.a ${TTMLIR_TOOLCHAIN_DIR}/lib
	DEPENDS stablehlo)

target_link_libraries(ttmlir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(ttmlir-opt ${LIBS})

set_property(GLOBAL PROPERTY STABLEHLO_LIBS "${STABLEHLO_LIBRARIES_LIST}")

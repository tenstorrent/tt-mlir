include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)

add_subdirectory(OpModel)
add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(Scheduler)

set(link_libs
MLIRTTDialect
MLIRTTTransforms
MLIRTTIRDialect
MLIRTTIRTransforms
TTMLIRConversions
MLIRTTNNAnalysis
MLIRTTNNDialect
MLIRTTNNTransforms
MLIRTTKernelDialect
MLIRTTMetalDialect
MLIRTTIRPipelines
MLIRTTNNPipelines
MLIRTTMetalPipelines
)

if (TTMLIR_ENABLE_STABLEHLO)
list(APPEND TTMLIR_LIBS
    StablehloBase
    StablehloReferenceIndex
    StablehloReferenceErrors
    StablehloReferenceElement
    StablehloReferenceAxes
    StablehloReferenceValue
    StablehloReferenceTypes
    StablehloReferenceToken
    StablehloReferenceTensor
    StablehloReferenceScope
    StablehloReferenceProcessGrid
    StablehloReferenceProcess
    StablehloReferenceOps
    StablehloPasses
    ChloOps
    VhloOps
    VhloTypes
    StablehloOps
    StablehloRegister
    StablehloLinalgTransforms
    StablehloSerialization
    InterpreterOps
    StablehloPortableApi
    StablehloAssemblyFormat
    StablehloBroadcastUtils
    StablehloTypeInference
    StablehloReferenceNumPy
    SdyDialect
    SdyRegister
)
endif()

add_library(TTMLIRCompiler SHARED RegisterAll.cpp)

add_dependencies(TTMLIRCompiler
    ${TTMLIR_LIBS}
)

# Forces the inclusion of all symbols in the shared object
# This is necessary because the JIT will not be able to find the symbols otherwise
if (APPLE)
  set(REEXPORT_TTMLIR_LIBS -Wl,-all_load ${TTMLIR_LIBS})
else()
  set(REEXPORT_TTMLIR_LIBS -Wl,--whole-archive ${TTMLIR_LIBS} -Wl,--no-whole-archive)
endif()

target_link_libraries(TTMLIRCompiler PRIVATE
    LLVM
    MLIR
    flatbuffers
    ${REEXPORT_TTMLIR_LIBS}
)

target_link_directories(TTMLIRCompiler PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib)

set_target_properties(TTMLIRCompiler PROPERTIES INSTALL_RPATH "$ORIGIN")

install(TARGETS TTMLIRCompiler DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT SharedLib)

set(TTMLIR_COMPILER_DEPS ${TTMLIR_TOOLCHAIN_DIR}/lib/libLLVM.so ${TTMLIR_TOOLCHAIN_DIR}/lib/libMLIR.so)
install(CODE
    "
    MESSAGE(STATUS \"Installing tt-mlir runtime shared lib dependencies (${TTMLIR_COMPILER_DEPS}) to ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
    file(
        INSTALL
        ${TTMLIR_COMPILER_DEPS}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
    )"
    COMPONENT SharedLib
    EXCLUDE_FROM_ALL
)

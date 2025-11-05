if (CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
endif()

include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)

add_subdirectory(OpModel)
add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(Scheduler)
add_subdirectory(Transforms)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(TTMLIR_LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ${translation_libs}
  MLIRPass
  MLIRSupport
  MLIRRegisterAllPasses
)

set(SHARDY_LIBS)
if (TTMLIR_ENABLE_STABLEHLO)
  list(APPEND SHARDY_LIBS
    SdyDialect
    SdyRegister
    SdyTransformsPropagationOpShardingRuleBuilder
    SdyTransformsPropagationPasses
    SdyTransformsPropagationOpShardingRuleRegistry
    SdyTransformsPropagationShardingGroupMap
    SdyTransformsPropagationShardingProjection
    SdyTransformsPropagationAutoPartitionerRegistry
    SdyAxisListRef
    SdyExplicitReshardsUtil
    SdyTransformsPropagationUtils
    SdyTransformsPropagationBasicFactorPropagation
    SdyTransformsPropagationAggressiveFactorPropagation
    SdyTransformsCommonOpProperties
    SdyTransformsCommonShardingWalker
    SdyTransformsPropagationDebuggingSourceSharding
    SdyCommonFileUtils
    SdyTransformsExportPasses
    SdyTransformsImportPasses
    SdyTransformsPasses
  )
endif()

set(SHARDY_HEADER_DIRS)
if (TTMLIR_ENABLE_STABLEHLO)
    list(APPEND SHARDY_HEADER_DIRS
      ${TTMLIR_TOOLCHAIN_DIR}/src/shardy
    )
endif()

#
# Static Target for tools like opt/translate/lspserver
#
add_mlir_library(TTMLIRCompilerStatic
  RegisterAll.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARDY_HEADER_DIRS}

  LINK_LIBS PUBLIC
  ${TTMLIR_LIBS}
  coverage_config
  ${SHARDY_LIBS}
)

#
# Shared Target for compiler frontends
#
add_library(TTMLIRCompiler SHARED RegisterAll.cpp)


set_target_properties(TTMLIRCompiler PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(TTMLIRCompiler PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

# Forces the inclusion of all symbols in the shared object
# This is necessary because the JIT will not be able to find the symbols otherwise
if (APPLE)
  set(REEXPORT_TTMLIR_LIBS -Wl,-all_load ${TTMLIR_LIBS})
else()
  set(REEXPORT_TTMLIR_LIBS -Wl,--whole-archive ${TTMLIR_LIBS} -Wl,--no-whole-archive)
endif()

target_link_libraries(TTMLIRCompiler PRIVATE
    ${REEXPORT_TTMLIR_LIBS}
    flatbuffers
)

target_link_libraries(TTMLIRCompiler PUBLIC coverage_config)

target_link_directories(TTMLIRCompiler PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib)
target_link_directories(TTMLIRCompiler PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib64)

install(TARGETS TTMLIRCompiler DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} COMPONENT SharedLib)

# Add coverage_config to all targets in lib
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target IN ITEMS TTMLIRCompilerStatic TTMLIRCompiler)
    if (TARGET ${target})
      target_link_libraries(${target} INTERFACE coverage_config)
    endif()
  endforeach()
endif()

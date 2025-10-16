message(STATUS "Runtimes Enabled: TTNN[${TT_RUNTIME_ENABLE_TTNN}] TTMETAL[${TT_RUNTIME_ENABLE_TTMETAL}]")

add_subdirectory(common)
add_subdirectory(ttnn)
add_subdirectory(ttmetal)

add_library(TTBinary STATIC binary.cpp)
set_property(TARGET TTBinary PROPERTY CXX_STANDARD 20)
target_include_directories(TTBinary
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
add_dependencies(TTBinary FBS_GENERATION)

add_library(TTMLIRRuntime SHARED runtime.cpp)
set_property(TARGET TTMLIRRuntime PROPERTY CXX_STANDARD 20)

target_include_directories(TTMLIRRuntime
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)

target_link_directories(TTMLIRRuntime PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib)
target_link_directories(TTMLIRRuntime PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib64)

target_link_libraries(TTMLIRRuntime
  PRIVATE
    TTBinary
    TTRuntimeTypes
    TTRuntimeUtils
    TTRuntimeSysDesc
    TTRuntimeTTNN
    TTRuntimeTTMetal
    TTRuntimeDistributed
    TTRuntimeContext
    TTRuntimeDebug
    TTRuntimePerf
    TTRuntimeWorkarounds
    TTRuntimeDylibs
    # This ensures that symbols from libTTRuntimeTTNNTestLib.a (libA) are linked into libTTMLIRRuntime.so (libB).
    # Since the symbols from libA aren't used in libB, linker will just ignore them. By using --whole-archive,
    # we tell linker to link all the symbols anyway. We need this as symbols from libA might be used by whoever
    # imports libB.
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,TTRuntimeTTNNTestLib>"
    ${METAL_LIBS}
    flatbuffers
)

target_link_libraries(TTMLIRRuntime PUBLIC coverage_config)

set_target_properties(TTMLIRRuntime PROPERTIES INSTALL_RPATH "$ORIGIN;${TTMETAL_LIBRARY_DIR}")
set_target_properties(TTMLIRRuntime PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

add_dependencies(TTMLIRRuntime FBS_GENERATION)

if (TTMLIR_ENABLE_RUNTIME)
  set(TTMLIR_RUNTIME_PUBLIC_HEADERS
    "../include/tt/runtime/debug.h"
    "../include/tt/runtime/perf.h"
    "../include/tt/runtime/runtime.h"
    "../include/tt/runtime/tensor_cache.h"
    "../include/tt/runtime/types.h"
    "../include/tt/runtime/utils.h"
    "../include/tt/runtime/workarounds.h"
  )
  set_target_properties(TTMLIRRuntime PROPERTIES PUBLIC_HEADER "${TTMLIR_RUNTIME_PUBLIC_HEADERS}")
  install(TARGETS TTMLIRRuntime
    PUBLIC_HEADER
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt/runtime
      COMPONENT SharedLib
    LIBRARY
      DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
      COMPONENT SharedLib
  )
  install(DIRECTORY ${PROJECT_BINARY_DIR}/runtime/include/tt/runtime/flatbuffer
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt/runtime
    COMPONENT SharedLib
    FILES_MATCHING
    PATTERN "*.h"
  )
endif()

add_subdirectory(distributed)

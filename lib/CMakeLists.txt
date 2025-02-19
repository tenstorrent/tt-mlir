include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)

add_subdirectory(OpModel)
add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(Scheduler)

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
)

#
# Static Target for tools like opt/translate/lspserver
#
add_mlir_library(TTMLIRCompilerStatic
  RegisterAll.cpp

  LINK_LIBS PUBLIC
  ${TTMLIR_LIBS}
)

#
# Shared Target for compiler frontends
#
add_library(TTMLIRCompiler SHARED RegisterAll.cpp)

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

target_link_directories(TTMLIRCompiler PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib)

install(TARGETS TTMLIRCompiler DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} COMPONENT SharedLib)

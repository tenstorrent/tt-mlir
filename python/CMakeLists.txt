include(AddMLIRPython)

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(TTMLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttmlir")
set(TTMLIR_PYTHON_TEST_INFRA_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_infra")

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ttmlir.")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

declare_mlir_python_sources(TTMLIRPythonSources)
declare_mlir_python_sources(TTMLIRPythonExtensions)
declare_mlir_python_sources(TTMLIRPythonTestInfra)

declare_mlir_python_sources(TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTEnumBinding.td
  SOURCES dialects/tt.py
  DIALECT_NAME tt
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTIRBinding.td
  SOURCES dialects/ttir.py
  DIALECT_NAME ttir
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTKernelBinding.td
  SOURCES dialects/ttkernel.py
  DIALECT_NAME ttkernel
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTNNBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTNNEnumBinding.td
  SOURCES dialects/ttnn.py
  DIALECT_NAME ttnn
)

declare_mlir_python_sources(TTMLIRPythonSources.Overrides
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTMLIRPythonSources
  SOURCES overrides.py
)

declare_mlir_python_sources(TTMLIRPythonSources.OptimizerOverrides
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTMLIRPythonSources
  SOURCES optimizer_overrides.py
)

declare_mlir_python_sources(TTMLIRPythonSources.Passes
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTMLIRPythonSources
  SOURCES passes.py
)

declare_mlir_python_sources(TTMLIRPythonSources.Util
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTMLIRPythonSources
  SOURCES util.py
)

declare_mlir_python_sources(TTMLIRPythonTestInfra.TestInfra
  ROOT_DIR "${TTMLIR_PYTHON_TEST_INFRA_ROOT_DIR}"
  ADD_TO_PARENT TTMLIRPythonTestInfra
  SOURCES
    ttir_builder.py
    test_utils.py
)

declare_mlir_python_extension(TTMLIRPythonExtensions.Main
  MODULE_NAME _ttmlir
  ADD_TO_PARENT TTMLIRPythonExtensions
  SOURCES
    TTMLIRModule.cpp
    TTModule.cpp
    TTIRModule.cpp
    TTKernelModule.cpp
    TTNNModule.cpp
    Overrides.cpp
    OptimizerOverrides.cpp
    Passes.cpp
    Util.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPITransforms
    TTMLIRCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    TTMLIRStatic
    ${extension_libs}
)

set(TTMLIR_PYTHON_SOURCES
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.affine
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.tensor
  MLIRPythonSources.Dialects.linalg
  MLIRPythonSources.Dialects.ml_program
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.tosa
  MLIRPythonSources.Dialects.memref
  TTMLIRPythonSources
  TTMLIRPythonExtensions
  TTMLIRPythonTestInfra
)

add_mlir_python_common_capi_library(TTMLIRPythonCAPI
  INSTALL_COMPONENT TTMLIRPythonModules
  INSTALL_DESTINATION python_packages/ttmlir/_mlir_libs
  OUTPUT_DIRECTORY "${TTMLIR_PYTHON_PACKAGES_DIR}/ttmlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES ${TTMLIR_PYTHON_SOURCES}
)

add_mlir_python_modules(TTMLIRPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/ttmlir"
  INSTALL_PREFIX "python_packages/ttmlir"
  DECLARED_SOURCES ${TTMLIR_PYTHON_SOURCES}
  COMMON_CAPI_LINK_LIBS
    TTMLIRPythonCAPI
)

if(TTMLIR_ENABLE_SHARDON)
  set(TTSHARDON_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shardon")
  declare_mlir_python_sources(TTSHARDONPythonSources
    ROOT_DIR "${TTSHARDON_PYTHON_ROOT_DIR}"
    SOURCES
      shardon_ast.py
  )
  add_mlir_python_modules(TTSHARDONPythonModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/shardon"
    INSTALL_PREFIX "python_packages/shardon"
    DECLARED_SOURCES TTSHARDONPythonSources
  )
endif()

install(DIRECTORY ${CMAKE_BINARY_DIR}/python/dialects/ DESTINATION python/dialects COMPONENT Test EXCLUDE_FROM_ALL)
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/ DESTINATION python_packages COMPONENT Test EXCLUDE_FROM_ALL)

include(AddMLIRPython)

################################################################################
# SHLO (copied over from /opt/ttmlir-toolchain/src/stablehlo/stablehlo/integrations/python/CMakeLists.txt)
################################################################################

# set(STABLEHLO_PYTHON_ROOT_DIR "${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo/stablehlo/integrations/python")

# declare_mlir_python_sources(StablehloPythonSources)
# declare_mlir_python_sources(StablehloPythonSources.Dialects
#   ADD_TO_PARENT StablehloPythonSources
# )

# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT StablehloPythonSources.Dialects
#   ROOT_DIR "${STABLEHLO_PYTHON_ROOT_DIR}/mlir"
#   TD_FILE dialects/StablehloOps.td
#   SOURCES dialects/stablehlo.py
#   DIALECT_NAME stablehlo)

# declare_mlir_python_sources(VhloPythonSources)
# declare_mlir_python_sources(VhloPythonSources.Dialects
#   ADD_TO_PARENT VhloPythonSources
# )

# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT VhloPythonSources.Dialects
#   ROOT_DIR "${STABLEHLO_PYTHON_ROOT_DIR}/mlir"
#   TD_FILE dialects/VhloOps.td
#   SOURCES dialects/vhlo.py
#   DIALECT_NAME vhlo)


# ## The following options are used specifically for compiling nanobind codebase.
# add_compile_options(-Wno-error=cast-qual -Wno-error=nested-anon-types
#   -Wno-error=extra-semi -Wno-error=zero-length-array
#   -Wno-error=covered-switch-default)

# declare_mlir_python_sources(StablehloPythonExtensions)
# declare_mlir_python_extension(StablehloPythonExtensions.Main
#   MODULE_NAME _stablehlo
#   ADD_TO_PARENT StablehloPythonExtensions
#   ROOT_DIR "${STABLEHLO_PYTHON_ROOT_DIR}"
#   PYTHON_BINDINGS_LIBRARY nanobind
#   SOURCES
#     StablehloApi.cpp
#     StablehloModule.cpp
#   EMBED_CAPI_LINK_LIBS
#     StablehloCAPI
#   PRIVATE_LINK_LIBS
#     StablehloCAPI
#     LLVMSupport
# )

# declare_mlir_python_sources(VhloPythonExtensions)
# declare_mlir_python_extension(VhloPythonExtensions.Main
#   MODULE_NAME _vhlo
#   ADD_TO_PARENT VhloPythonExtensions
#   ROOT_DIR "${STABLEHLO_PYTHON_ROOT_DIR}"
#   PYTHON_BINDINGS_LIBRARY nanobind
#   SOURCES
#     VhloModule.cpp
#   EMBED_CAPI_LINK_LIBS
#     VhloCAPI
#   PRIVATE_LINK_LIBS
#     LLVMSupport
# )

################################################################################
# TTMLIR
################################################################################

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(TTMLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttmlir")
set(TTMLIR_PYTHON_INFRA_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/infra")

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ttmlir.")

declare_mlir_python_sources(TTMLIRPythonSources)
declare_mlir_python_sources(TTMLIRPythonExtensions)
declare_mlir_python_sources(TTMLIRPythonInfra)

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

declare_mlir_python_sources(TTMLIRPythonInfra.Infra
  ROOT_DIR "${TTMLIR_PYTHON_INFRA_ROOT_DIR}"
  ADD_TO_PARENT TTMLIRPythonInfra
  SOURCES
    compile_and_run.py
    mlir_module_executor.py
    mlir_module_splitter.py
    pydantic_models.py
    stablehlo_executor.py
    stablehlo_splitter.py
    test_optimizer_overrides.py
    test_utils.py
    ttir_builder.py
    ttir_executor.py
    ttir_splitter.py
    ttnn_executor.py
    ttnn_splitter.py
    utils.py
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
    TTMLIRCompilerStatic
    MLIRAMXToLLVMIRTranslation
    MLIRArmNeonToLLVMIRTranslation
    MLIRArmSMEToLLVMIRTranslation
    MLIRArmSVEToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    MLIRGPUToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    MLIRNVVMToLLVMIRTranslation
    MLIROpenACCToLLVMIRTranslation
    MLIROpenMPToLLVMIRTranslation
    MLIRROCDLToLLVMIRTranslation
    MLIRSPIRVToLLVMIRTranslation
    MLIRTestToLLVMIRTranslation
    MLIRVCIXToLLVMIRTranslation
    MLIRX86VectorToLLVMIRTranslation
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
  TTMLIRPythonInfra
)

if(TTMLIR_ENABLE_STABLEHLO)
  # The following options are used specifically for compiling nanobind codebase.
  add_compile_options(-Wno-error=cast-qual -Wno-error=nested-anon-types
    -Wno-error=extra-semi -Wno-error=zero-length-array
    -Wno-error=covered-switch-default)

  list(APPEND TTMLIR_PYTHON_SOURCES StablehloPythonExtensions)
endif()

################################################################################
# SHLO
################################################################################

# add_mlir_python_common_capi_library(StablehloUnifiedPythonCAPI
#   INSTALL_COMPONENT StablehloUnifiedPythonModules
#   INSTALL_DESTINATION python_packages/ttmlir/_mlir_libs
#   OUTPUT_DIRECTORY "${TTMLIR_PYTHON_PACKAGES_DIR}/ttmlir/_mlir_libs"
#   RELATIVE_INSTALL_ROOT ".."
#   DECLARED_SOURCES
#     StablehloPythonSources
#     StablehloPythonExtensions
#     VhloPythonSources
#     VhloPythonExtensions
# )

# add_mlir_python_modules(StablehloUnifiedPythonModules
#   ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/ttmlir"
#   INSTALL_PREFIX "python_packages/ttmlir"
#   DECLARED_SOURCES
#     StablehloPythonSources
#     StablehloPythonExtensions
#     VhloPythonSources
#     VhloPythonExtensions
#   COMMON_CAPI_LINK_LIBS
#     StablehloUnifiedPythonCAPI
# )

################################################################################
# TTMLIR
################################################################################

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

if(TTMLIR_ENABLE_PYKERNEL)
  set(TTPYKERNEL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pykernel")
  declare_mlir_python_sources(TTPykernelSources
    ROOT_DIR "${TTPYKERNEL_ROOT_DIR}"
    SOURCES
      pykernel_ast.py
      types.py
  )
  add_mlir_python_modules(TTPykernelModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/pykernel"
    INSTALL_PREFIX "python_packages/pykernel"
    DECLARED_SOURCES TTPykernelSources
  )
endif()

install(DIRECTORY ${CMAKE_BINARY_DIR}/python/dialects/ DESTINATION python/dialects COMPONENT Test EXCLUDE_FROM_ALL)
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/ DESTINATION python_packages COMPONENT Test EXCLUDE_FROM_ALL)

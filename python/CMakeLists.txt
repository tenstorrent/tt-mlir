include(AddMLIRPython)

# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(TTMLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttmlir")

# We vendor our own MLIR instance in the `ttmlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ttmlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(TTMLIRPythonSources)

declare_mlir_python_sources(TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTCoreBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTCoreEnumBinding.td
  SOURCES dialects/ttcore.py
  DIALECT_NAME ttcore
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
  TD_FILE dialects/D2MBinding.td
  SOURCES dialects/d2m.py
  DIALECT_NAME d2m
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTKernelBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTKernelEnumBinding.td
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

# Enable Site Initialization Target to be piped into _mlir_libs/__init__.py
declare_mlir_python_sources(TTMLIRPythonSiteInitialize
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTMLIRPythonSources
  SOURCES _mlir_libs/_site_initialize_0.py
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_sources(TTMLIRPythonExtensions)

declare_mlir_python_extension(TTMLIRPythonExtensions.Main
  MODULE_NAME _ttmlir
  ADD_TO_PARENT TTMLIRPythonExtensions
  SOURCES
    TTMLIRModule.cpp
    TTModule.cpp
    TTKernelModule.cpp
    TTNNModule.cpp
    OptimizerOverrides.cpp
    Passes.cpp
    Util.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPITransforms
    TTMLIRCAPI
  PRIVATE_LINK_LIBS
    TTMLIRCompilerStatic
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
    MLIRX86VectorTransforms
  PYTHON_BINDINGS_LIBRARY nanobind
)

set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "ttmlir")

################################################################################
# Additional custom python code we want to bundle in the package
################################################################################

declare_mlir_python_sources(TTMLIRPythonCommon)

declare_mlir_python_sources(TTMLIRPythonCommon.Main
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common"
  ADD_TO_PARENT TTMLIRPythonCommon
  SOURCES
    compile_and_run_internal.py
    compile_and_run_utils.py
    compile_and_run.py
)

################################################################################
# Generate packages and shared library
################################################################################

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
  MLIRPythonSources.Dialects.emitc
  MLIRPythonSources.Dialects.quant
  TTMLIRPythonSources
  TTMLIRPythonExtensions
  TTMLIRPythonCommon
  TTMLIRPythonSiteInitialize
)

if(TTMLIR_ENABLE_STABLEHLO)
  # The following options are used specifically for compiling nanobind codebase.
  # They are copied over from SHLO's cmake file and will prevent nanobind warnings
  # being treated as build errors and breaking the build.
  add_compile_options(-Wno-error=cast-qual -Wno-error=nested-anon-types
    -Wno-error=extra-semi -Wno-error=zero-length-array
    -Wno-error=covered-switch-default)

  # To defined python sources we append minimal set of targets exposed by SHLO's cmake.
  # This will put `stablehlo` under `ttmlir` and allow us to import it from `ttmlir`.
  list(APPEND TTMLIR_PYTHON_SOURCES
    StablehloPythonSources
    StablehloPythonExtensions
    SdyPythonSources
    SdyPythonExtensions
    # https://github.com/tenstorrent/tt-mlir/issues/5281
    # MpmdPythonSources
    # MpmdPythonExtensions
  )
endif()

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

# PyWheel Component to create install directives for the packaged TTMLIR bindings wheel:
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/ DESTINATION . COMPONENT TTMLIRPythonWheel EXCLUDE_FROM_ALL)

install(DIRECTORY ${CMAKE_BINARY_DIR}/python/dialects/ DESTINATION python/dialects COMPONENT Test EXCLUDE_FROM_ALL)
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/ DESTINATION python_packages COMPONENT Test EXCLUDE_FROM_ALL)

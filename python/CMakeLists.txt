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
  DEPENDS ${CMAKE_SOURCE_DIR}/include/ttmlir/Dialect/TTCore/IR/TTCoreOps.td
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTIRBinding.td
  SOURCES dialects/ttir.py
  DIALECT_NAME ttir
  DEPENDS ${CMAKE_SOURCE_DIR}/include/ttmlir/Dialect/TTIR/IR/TTIROps.td
          ${CMAKE_SOURCE_DIR}/include/ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.td
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/D2MBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/D2MEnumBinding.td
  SOURCES dialects/d2m.py
  DIALECT_NAME d2m
  DEPENDS ${CMAKE_SOURCE_DIR}/include/ttmlir/Dialect/D2M/IR/D2MOps.td
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTKernelBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTKernelEnumBinding.td
  SOURCES dialects/ttkernel.py
  DIALECT_NAME ttkernel
  DEPENDS ${CMAKE_SOURCE_DIR}/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTNNBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTNNEnumBinding.td
  SOURCES dialects/ttnn.py
  DIALECT_NAME ttnn
  DEPENDS ${CMAKE_SOURCE_DIR}/include/ttmlir/Dialect/TTNN/IR/TTNNOps.td
          ${CMAKE_SOURCE_DIR}/include/ttmlir/Dialect/TTNN/IR/TTNNOpsInterfaces.td
)

# Generate a Python op-schema sidecar (operand/attr/result names) from the
# TTNN tablegen JSON dump. Produces _ttnn_op_schema.py next to the existing
# _ttnn_ops_gen.py and packages it as part of TTMLIRPythonSources.
set(_ttnn_op_schema_py "${CMAKE_CURRENT_BINARY_DIR}/dialects/_ttnn_op_schema.py")
set(_ttnn_ops_json "${CMAKE_BINARY_DIR}/include/ttmlir/Dialect/TTNN/IR/TTNNOps.json")
set(_op_schema_gen "${CMAKE_SOURCE_DIR}/tools/scripts/python_op_schema_codegen.py")
add_custom_command(
  OUTPUT "${_ttnn_op_schema_py}"
  COMMAND ${Python3_EXECUTABLE} "${_op_schema_gen}"
            --json "${_ttnn_ops_json}"
            --out "${_ttnn_op_schema_py}"
  DEPENDS "${_ttnn_ops_json}" "${_op_schema_gen}" MLIRTTNNOpsJsonIncGen
  COMMENT "Building dialects/_ttnn_op_schema.py..."
)
declare_mlir_python_sources(TTMLIRPythonSources.Dialects.ttnn.op_schema
  ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  ADD_TO_PARENT TTMLIRPythonSources.Dialects.ttnn
  SOURCES "dialects/_ttnn_op_schema.py"
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/DebugBinding.td
  SOURCES dialects/debug.py
  DIALECT_NAME debug
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
    TTIRModule.cpp
    D2MModule.cpp
    TTKernelModule.cpp
    TTNNModule.cpp
    DebugModule.cpp
    OptimizerOverrides.cpp
    Passes.cpp
    Util.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPITransforms
    TTMLIRCAPI
)

# Create a separate library for PassTracker compiled with -fno-rtti to match MLIR
add_library(TTMLIRPassTracker STATIC PassTracker.cpp)
target_compile_options(TTMLIRPassTracker PRIVATE -fno-rtti)
target_include_directories(TTMLIRPassTracker PRIVATE ${MLIR_INCLUDE_DIRS})
target_link_libraries(TTMLIRPassTracker PRIVATE MLIRPass MLIRIR LLVMSupport)

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
    _flatbuffer_worker.py
    _system_desc_worker.py
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
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.complex
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
    MpmdPythonSources
    MpmdPythonExtensions
  )
endif()

add_mlir_python_common_capi_library(TTMLIRPythonCAPI
  INSTALL_COMPONENT TTMLIRPythonModules
  INSTALL_DESTINATION python_packages/ttmlir/_mlir_libs
  OUTPUT_DIRECTORY "${TTMLIR_PYTHON_PACKAGES_DIR}/ttmlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES ${TTMLIR_PYTHON_SOURCES}
)
# The python extensions must stay thin (no embedded MLIR/TTMLIR statics:
# duplicate storage uniquers across .so files crash with "storage uniquer
# isn't initialized"). Their MLIR C-API/PassTracker needs are force-extracted
# into the CAPI here; identical flags in separate target_link_options calls
# get deduped by CMake, so keep ONE whole-archive block.
set(TTMLIR_CAPI_FORCE_EXTRACT
  "$<TARGET_FILE:TTMLIRPassTracker>"
  "$<TARGET_FILE:TTMLIRTTNNUtils>"
  "$<TARGET_FILE:TTMLIRCompilerStatic>"
  "$<TARGET_FILE:MLIRTTNNDialect>"
  "$<TARGET_FILE:MLIRTTCoreDialect>"
  "$<TARGET_FILE:MLIRTTKernelDialect>"
  "$<TARGET_FILE:MLIRD2MDialect>"
  "$<TARGET_FILE:MLIRTTIRDialect>"
  "$<TARGET_FILE:MLIRTTNNTransforms>"
  "$<TARGET_FILE:MLIRTTNNAnalysis>"
  "$<TARGET_FILE:MLIRTTNNPipelines>"
  "$<TARGET_FILE:MLIRTTUtils>"
  "$<TARGET_FILE:MLIRD2MAnalysis>"
  "$<TARGET_FILE:MLIRD2MTransforms>"
  "$<TARGET_FILE:MLIRD2MPipelines>"
  "$<TARGET_FILE:MLIRStableHLOTransforms>"
  "$<TARGET_FILE:TTLLVMToDynamicLib>"
  "$<TARGET_FILE:TTNNTargetFlatbuffer>"
  "$<TARGET_FILE:TTKernelTargetCpp>"
  "$<TARGET_FILE:TTMetalTargetFlatbuffer>"
  "$<TARGET_FILE:EmitPyTargetPython>"
  "$<TARGET_FILE:CheckOps>"
  "$<TARGET_FILE:MpmdTransformsCommonUtils>"
  # MLIR-side owners referenced by the ttmlir bindings but never pulled by the
  # CAPI's own code.
  "$<TARGET_FILE:MLIRRegisterAllPasses>"
  "$<TARGET_FILE:MLIRGPUToLLVMIRTranslation>"
  "$<TARGET_FILE:MLIRXeVMToLLVMIRTranslation>"
  "$<TARGET_FILE:MLIRVCIXToLLVMIRTranslation>"
  "$<TARGET_FILE:MLIRTargetCpp>"
  "$<TARGET_FILE:MLIRSPIRVToLLVMIRTranslation>"
  "$<TARGET_FILE:MLIRROCDLToLLVMIRTranslation>"
  "$<TARGET_FILE:MLIRPtrToLLVMIRTranslation>"
  "$<TARGET_FILE:MLIROpenMPToLLVMIRTranslation>"
  "$<TARGET_FILE:MLIROpenACCToLLVMIRTranslation>")
if(TARGET SdyAxisListRef)
  # Shardy at the pinned SHA lists SdyAxisListRef before SdyExplicitReshardsUtil
  # in SdyTransformsExportPasses' LINK_LIBS, so single-pass ld leaves AxisListRef
  # undefined when the CAPI .so is linked with -Wl,-z,defs. Remove when shardy
  # fixes its link order.
  list(APPEND TTMLIR_CAPI_FORCE_EXTRACT "$<TARGET_FILE:SdyAxisListRef>")
endif()
target_link_options(TTMLIRPythonCAPI PRIVATE
  "-Wl,--whole-archive"
  ${TTMLIR_CAPI_FORCE_EXTRACT}
  "-Wl,--no-whole-archive"
  # --gc-sections drops weak TypeID objects (e.g. CoreRangeSetAttr) that are
  # only referenced by the python extensions, leaving them undefined at
  # runtime; keep all pulled sections in the CAPI.
  "-Wl,--no-gc-sections")

add_mlir_python_modules(TTMLIRPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/ttmlir"
  INSTALL_PREFIX "python_packages/ttmlir"
  DECLARED_SOURCES ${TTMLIR_PYTHON_SOURCES}
  COMMON_CAPI_LINK_LIBS
    TTMLIRPythonCAPI
)

# HACK: Use flat namespace on macOS to allow RTTI symbols from MLIRPass/LLVMSupport
# to be found at runtime. This is needed because PassInstrumentation RTTI isn't
# exported by default from the pre-built MLIR libraries.
if(APPLE)
  set_property(TARGET TTMLIRPythonModules.extension._ttmlir.dso
    APPEND PROPERTY LINK_FLAGS "-Wl,-flat_namespace,-undefined,dynamic_lookup")
endif()

# PyWheel Component to create install directives for the packaged TTMLIR bindings wheel:
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/ttmlir DESTINATION . COMPONENT TTMLIRPythonWheel EXCLUDE_FROM_ALL)

install(DIRECTORY ${CMAKE_BINARY_DIR}/python/dialects/ DESTINATION python/dialects COMPONENT Test EXCLUDE_FROM_ALL)
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/ DESTINATION python_packages COMPONENT Test EXCLUDE_FROM_ALL)

# TTMLIRInstall.cmake
# Install configuration for TTMLIR targets.
# This module explicitly lists all TTMLIR targets for export.

# Explicitly list all TTMLIR targets that should be exported.
# This avoids accidentally including third-party targets (StableHLO, Shardy, etc.)
# that may be collected via MLIR property lists.
# TODO: A better solution would be to define a set of cmake functions for
# registering TTMLIR targets that set the proper properties for export.
set(ttmlir_export_targets
  # Core dialect and utilities
  MLIRTTCoreDialect
  MLIRTTTransforms
  MLIRTTUtils

  # TTIR dialect
  MLIRTTIRDialect
  MLIRTTIREraseInverseOps
  MLIRTTIRTransforms
  TTMLIRTTIRUtils

  # TTNN dialect
  TTMLIRTTNNUtils
  MLIRTTNNDialect
  MLIRTTNNAnalysis
  MLIRTTNNTransforms
  MLIRTTNNValidation
  MLIRTTNNInterfaces

  # TTMetal dialect
  MLIRTTMetalDialect
  MLIRTTMetalPipelines

  # TTKernel dialect
  MLIRTTKernelDialect
  MLIRTTKernelPipelines
  MLIRTTKernelTransforms

  # Other dialects
  MLIRDebugDialect
  MLIRSFPIDialect
  MLIRLLVMTransforms
  MLIREmitPyDialect
  MLIREmitPyInterfaces
  MLIRD2MDialect
  MLIRD2MAllocation
  MLIRD2MAnalysis
  MLIRD2MTransforms
  MLIRD2MUtils

  # Conversions
  TTMLIRTosaToTTIR
  TTMLIRTTIRToTTIRDecomposition
  TTMLIRD2MToTTNN
  TTMLIRTTIRToD2M
  TTMLIRArithToD2MTileOps
  TTMLIRMathToD2MTileOps
  TTMLIRTTNNToEmitC
  TTMLIRTTNNToEmitPy
  TTMLIRTTIRToLinalg
  TTMLIRTTIRToTTNN
  TTMLIRTTKernelToEmitC
  TTMLIRD2MToTTKernel
  TTMLIRD2MToTTMetal
  TTMLIRSFPIToEmitC
  TTMLIRTTNNToTTIR

  # Transforms
  TTMLIRTransforms

  # Targets
  TTMetalTargetFlatbuffer
  TTNNTargetFlatbuffer
  TTLLVMToDynamicLib
  TTKernelTargetCpp
  EmitPyTargetPython

  # Utilities and support
  TTMLIRSupport
  coverage_config
  TTNNOpModelLib
  MLIRScheduler
)

# When TTMLIR_ENABLE_STABLEHLO=ON, certain targets link against third-party
# StableHLO/Shardy libraries that we don't want to export as part of TTMLIR.
# Therefore, we only export these targets when StableHLO is disabled (they
# become stubs or have no external dependencies in that configuration).
if(NOT TTMLIR_ENABLE_STABLEHLO)
  list(APPEND ttmlir_export_targets
    # TTIR pipelines (when StableHLO is enabled, depends on third-party StableHLO libs)
    MLIRTTIRPipelines

    # TTNN pipelines (depends on TTIR pipelines)
    MLIRTTNNPipelines

    # StableHLO integration (when StableHLO is enabled, depends on third-party Shardy libs)
    MLIRStableHLOTransforms
    MLIRStableHLOPipelines
    TTMLIRStableHLOUtils
    TTMLIRStableHLOToTTIR
  )
endif()

# Filter to only valid, non-imported targets that actually exist
set(ttmlir_export_targets_filtered)
foreach(target IN LISTS ttmlir_export_targets)
  if(TARGET ${target})
    get_target_property(target_imported ${target} IMPORTED)
    if(NOT target_imported)
      list(APPEND ttmlir_export_targets_filtered ${target})
    endif()
  endif()
endforeach()
list(REMOVE_DUPLICATES ttmlir_export_targets_filtered)

# Set global property for use in CMakeLists.txt
set_property(GLOBAL PROPERTY TTMLIR_EXPORTS ${ttmlir_export_targets_filtered})

set(TTMLIR_TTRT_WHEEL_REQUIRES_DIST "torch==2.3.0" CACHE STRING "Requires-Dist entries for ttrt wheel")
set_property(GLOBAL PROPERTY TTMLIR_TTRT_WHEEL_REQUIRES_DIST "${TTMLIR_TTRT_WHEEL_REQUIRES_DIST}")
if(NOT DEFINED TTMLIR_TTRT_WHEEL_REQUIRES_DIST_METADATA)
  set(TTMLIR_TTRT_WHEEL_REQUIRES_DIST_METADATA "Requires-Dist: ${TTMLIR_TTRT_WHEEL_REQUIRES_DIST}")
endif()
set(TTMLIR_TTRT_WHEEL_REQUIRES_DIST_METADATA "${TTMLIR_TTRT_WHEEL_REQUIRES_DIST_METADATA}" CACHE STRING "Requires-Dist metadata lines for ttrt wheel" FORCE)
set_property(GLOBAL PROPERTY TTMLIR_TTRT_WHEEL_REQUIRES_DIST_METADATA "${TTMLIR_TTRT_WHEEL_REQUIRES_DIST_METADATA}")

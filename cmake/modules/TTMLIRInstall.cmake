# TTMLIRInstall.cmake
# Install configuration for TTMLIR targets.
# This module collects and dedups all TTMLIR targets for export.

# Collect all TTMLIR library targets from MLIR properties.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# Collect all TTMLIR targets that should be exported.
set(ttmlir_export_targets)
list(APPEND ttmlir_export_targets ${dialect_libs})
list(APPEND ttmlir_export_targets ${conversion_libs})
list(APPEND ttmlir_export_targets ${extension_libs})
list(APPEND ttmlir_export_targets ${translation_libs})

# Add explicitly needed targets that may not be in property lists.
list(APPEND ttmlir_export_targets
  coverage_config
  TTNNOpModelLib
  MLIRTTNNInterfaces
  MLIRScheduler
  MLIRD2MUtils
)

# Remove duplicates and filter to only valid, non-imported targets.
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

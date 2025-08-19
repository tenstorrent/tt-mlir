add_subdirectory(Common)
add_subdirectory(TTMetal)
add_subdirectory(TTNN)
add_subdirectory(TTKernel)

add_custom_target(FBS_GENERATION DEPENDS TTMETAL_FBS TTNN_FBS COMMON_FBS)

if (TTMLIR_ENABLE_PYKERNEL)
  add_dependencies(COMMON_FBS_PY TTMETAL_FBS_PY TTNN_FBS_PY)
  add_custom_target(FBS_GENERATION_PY DEPENDS COMMON_FBS_PY)
  add_dependencies(mlir-headers FBS_GENERATION_PY)
endif()

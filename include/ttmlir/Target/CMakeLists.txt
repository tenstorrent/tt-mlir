add_subdirectory(Common)
add_subdirectory(TTMetal)
add_subdirectory(TTNN)
add_subdirectory(TTKernel)
add_subdirectory(CUDA)

add_custom_target(FBS_GENERATION DEPENDS TTMETAL_FBS TTNN_FBS COMMON_FBS CUDA_FBS)

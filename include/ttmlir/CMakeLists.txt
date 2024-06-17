add_subdirectory(Dialect)
add_subdirectory(Target)
add_dependencies(mlir-headers TTMETAL_FBS TTNN_FBS COMMON_FBS)

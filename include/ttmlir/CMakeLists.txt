add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(Transforms)

# if (TTMLIR_ENABLE_RUNTIME)
#    add_subdirectory(Runtime)
#endif()

add_dependencies(mlir-headers FBS_GENERATION)

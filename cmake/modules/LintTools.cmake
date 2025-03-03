# clang-tidy setup
add_custom_target(clang-tidy COMMAND run-clang-tidy.py -p ${PROJECT_BINARY_DIR} -export-fixes clang-tidy-fixes.yaml -warnings-as-errors '*' -extra-arg-before=-DDISABLE_STATIC_ASSERT_TESTS -extra-arg-before=-D__cpp_structured_bindings=202400
  DEPENDS
    mlir-headers
    mlir-generic-headers
    tt-metal-download
    tt-metal-configure
    FBS_GENERATION
)
add_custom_target(clang-format COMMAND git-clang-format)

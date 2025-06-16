# clang-tidy setup
add_custom_target(clang-tidy COMMAND run-clang-tidy.py -p ${PROJECT_BINARY_DIR} -export-fixes clang-tidy-fixes.yaml -warnings-as-errors '*' -extra-arg-before=-DDISABLE_STATIC_ASSERT_TESTS -extra-arg-before=-D__cpp_structured_bindings=202400
  DEPENDS
    mlir-headers
    mlir-generic-headers
    tt-metal-download
    tt-metal-configure
    FBS_GENERATION
)

# clang-tidy setup for CI run
add_custom_target(clang-tidy-ci
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/scripts/filter-compile-commands.py --prefix ${CMAKE_SOURCE_DIR} --diff ${CMAKE_BINARY_DIR}/compile_commands.json
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} -- clang-tidy
  COMMENT "Running clang-tidy CI checks"
  DEPENDS
    FBS_GENERATION
    mlir-headers
    mlir-generic-headers
    tt-metal-download
    tt-metal-update
    tt-metal-configure
    TTKernelGeneratedLLKHeaders
)

add_custom_target(clang-format COMMAND git-clang-format)

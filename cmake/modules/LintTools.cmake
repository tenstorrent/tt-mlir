# clang-tidy setup
set(CLANG_TIDY_DEPS
  mlir-headers
  mlir-generic-headers
  tt-metal-download
  tt-metal-configure
  FBS_GENERATION
  RUNTIME_FBS_GENERATION
)

set(CLANG_TIDY_CI_DEPS
  FBS_GENERATION
  RUNTIME_FBS_GENERATION
  mlir-headers
  mlir-generic-headers
  tt-metal-download
  tt-metal-update
  tt-metal-configure
  TTKernelGeneratedLLKHeaders
)

# Add StableHLO dependencies when enabled
if(TTMLIR_ENABLE_STABLEHLO)
  list(APPEND CLANG_TIDY_DEPS
    StablehloLinalgTransformsPassIncGen  # This generates the .inc file you need
  )
  list(APPEND CLANG_TIDY_CI_DEPS
    StablehloLinalgTransformsPassIncGen  # This generates the .inc file you need
  )
endif()

add_custom_target(clang-tidy
  COMMAND run-clang-tidy.py -p ${PROJECT_BINARY_DIR} -export-fixes clang-tidy-fixes.yaml -warnings-as-errors '*' -extra-arg-before=-DDISABLE_STATIC_ASSERT_TESTS -extra-arg-before=-D__cpp_structured_bindings=202400
  DEPENDS ${CLANG_TIDY_DEPS}
)

# clang-tidy setup for CI run
add_custom_target(clang-tidy-ci
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/scripts/filter-compile-commands.py --prefix ${CMAKE_SOURCE_DIR} --diff ${CMAKE_BINARY_DIR}/compile_commands.json
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} -- clang-tidy
  COMMENT "Running clang-tidy CI checks"
  DEPENDS ${CLANG_TIDY_CI_DEPS}
)

add_custom_target(clang-format COMMAND git-clang-format)

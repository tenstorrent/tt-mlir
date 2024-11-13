# clang-tidy setup
add_custom_target(clang-tidy-filter-out-external-srcs COMMAND python3 ${TTMLIR_SOURCE_DIR}/tools/scripts/filter-compile-commands.py ${TTMLIR_BINARY_DIR}/compile_commands.json "${TTMLIR_SOURCE_DIR}")
add_custom_target(clang-tidy COMMAND run-clang-tidy.py -p ${PROJECT_BINARY_DIR} -warnings-as-errors '*' -extra-arg-before=-DDISABLE_STATIC_ASSERT_TESTS -extra-arg-before=-D__cpp_structured_bindings=202400 DEPENDS clang-tidy-filter-out-external-srcs)
add_custom_target(clang-format COMMAND git-clang-format)

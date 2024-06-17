# clang-tidy setup
add_custom_target(clang-tidy-filter-out-external-srcs COMMAND python3 ${TTMLIR_SOURCE_DIR}/tools/scripts/filter-compile-commands.py ${TTMLIR_BINARY_DIR}/compile_commands.json ${TTMLIR_SOURCE_DIR})
add_custom_target(clang-tidy COMMAND run-clang-tidy -p ${PROJECT_BINARY_DIR} -warnings-as-errors '*' DEPENDS clang-tidy-filter-out-external-srcs)
add_custom_target(clang-format COMMAND git-clang-format)

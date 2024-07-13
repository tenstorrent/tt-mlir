# clang-tidy setup
add_custom_target(run-clang-tidy-install COMMAND cp ${TTMLIR_TOOLCHAIN_DIR}/src/llvm-project/clang-tools-extra/clang-tidy/tool/run-clang-tidy.py ${TTMLIR_TOOLCHAIN_DIR}/bin/)
add_custom_target(clang-tidy-filter-out-external-srcs COMMAND python3 ${TTMLIR_SOURCE_DIR}/tools/scripts/filter-compile-commands.py ${TTMLIR_BINARY_DIR}/compile_commands.json ${TTMLIR_SOURCE_DIR})
add_custom_target(clang-tidy COMMAND run-clang-tidy.py -p ${PROJECT_BINARY_DIR} -warnings-as-errors '*' DEPENDS clang-tidy-filter-out-external-srcs run-clang-tidy-install)
add_custom_target(clang-format COMMAND git-clang-format)

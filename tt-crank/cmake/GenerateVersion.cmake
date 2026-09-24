include("${CMAKE_CURRENT_LIST_DIR}/GitRev.cmake")

git_commit_hash(TTMLIR_GIT_HASH "${TTMLIR_SOURCE_DIR}")
if(TTMLIR_GIT_HASH STREQUAL "")
    message(FATAL_ERROR
        "Could not resolve the tt-mlir commit hash from "
        "${TTMLIR_SOURCE_DIR}.")
endif()

git_scoped_worktree_hash(
    TTMLIR_GIT_WORKTREE_HASH
    "${TTMLIR_SOURCE_DIR}"
    lib
    include
)
if(TTMLIR_GIT_WORKTREE_HASH STREQUAL "")
    message(FATAL_ERROR
        "Could not resolve the tt-mlir compiler worktree hash from "
        "${TTMLIR_SOURCE_DIR}.")
endif()

git_worktree_hash(TTMETAL_GIT_WORKTREE_HASH "${TTMETAL_SOURCE_DIR}")
if(TTMETAL_GIT_WORKTREE_HASH STREQUAL "")
    message(FATAL_ERROR
        "Could not resolve the tt-metal worktree hash from "
        "${TTMETAL_SOURCE_DIR}.")
endif()

configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/../src/version.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/version.cpp"
    @ONLY
)

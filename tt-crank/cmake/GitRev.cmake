# git_commit_hash(<out_var> <source_dir>)
# Returns the HEAD commit hash of the git checkout containing <source_dir> via
# <out_var>. Sets <out_var> to an empty string and warns when <source_dir> is
# not a git checkout.
function(git_commit_hash out_var source_dir)
    execute_process(
        COMMAND git -C ${source_dir} rev-parse HEAD
        OUTPUT_VARIABLE _hash
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rc
        ERROR_QUIET
    )

    if(NOT _rc EQUAL 0)
        message(WARNING "git_commit_hash: 'git rev-parse HEAD' failed in "
                        "${source_dir}; returning empty string")
        set(_hash "")
    endif()

    set(${out_var} "${_hash}" PARENT_SCOPE)
endfunction()

# git_worktree_hash(<out_var> <source_dir>)
# Hashes the current commit and tracked local changes in <source_dir>.
# Untracked files are ignored.
function(git_worktree_hash out_var source_dir)
    git_commit_hash(_base "${source_dir}")
    if(_base STREQUAL "")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    # --no-color so the digest doesn't depend on the caller's git color config.
    execute_process(
        COMMAND git -C ${source_dir} diff --no-color HEAD
        OUTPUT_VARIABLE _diff
        RESULT_VARIABLE _rc
        ERROR_QUIET
    )
    if(NOT _rc EQUAL 0)
        message(WARNING "git_worktree_hash: 'git diff HEAD' failed in "
                        "${source_dir}; returning empty string")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    if(_diff STREQUAL "")
        set(${out_var} "${_base}" PARENT_SCOPE)
        return()
    endif()

    string(SHA1 _combined "${_base}${_diff}")
    set(${out_var} "${_combined}" PARENT_SCOPE)
endfunction()

# git_scoped_worktree_hash(<out_var> <source_dir> <path>...)
# Hashes the committed contents and tracked local changes under the given
# paths (unlike `git_worktree_hash` which takes in the whole repo).
# Changes outside those paths and untracked files are ignored.
function(git_scoped_worktree_hash out_var source_dir)
    set(_paths ${ARGN})
    if(NOT _paths)
        message(WARNING "git_scoped_worktree_hash: no paths were provided")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    set(_revisions)
    foreach(_path IN LISTS _paths)
        list(APPEND _revisions "HEAD:${_path}")
    endforeach()

    execute_process(
        COMMAND git -C "${source_dir}" rev-parse ${_revisions}
        OUTPUT_VARIABLE _objects
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rc
        ERROR_QUIET
    )
    if(NOT _rc EQUAL 0 OR _objects STREQUAL "")
        message(WARNING "git_scoped_worktree_hash: 'git rev-parse' failed "
                        "or returned no entries in ${source_dir}; returning empty string")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    string(SHA1 _base "${_objects}")

    # --no-color so the digest doesn't depend on the caller's git color config.
    execute_process(
        COMMAND git -C "${source_dir}" diff --no-color HEAD -- ${_paths}
        OUTPUT_VARIABLE _diff
        RESULT_VARIABLE _rc
        ERROR_QUIET
    )
    if(NOT _rc EQUAL 0)
        message(WARNING "git_scoped_worktree_hash: 'git diff HEAD' failed in "
                        "${source_dir}; returning empty string")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    if(_diff STREQUAL "")
        set(${out_var} "${_base}" PARENT_SCOPE)
        return()
    endif()

    string(SHA1 _combined "${_base}${_diff}")
    set(${out_var} "${_combined}" PARENT_SCOPE)
endfunction()

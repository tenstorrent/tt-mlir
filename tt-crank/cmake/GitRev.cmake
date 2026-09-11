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
# Single identifier combining the HEAD commit with any uncommitted *tracked*
# changes (staged + unstaged) in <source_dir>: the plain commit hash when the
# tree is clean, and SHA-1(commit + `git diff HEAD`) when there are local edits.
# So the value is stable across clean builds but changes whenever a local edit
# changes what feeds the build — suitable as an on-disk cache key. Empty string
# (with a warning) when <source_dir> is not a git checkout or `git diff HEAD`
# fails. Untracked files are not included (they only affect a build once
# referenced from a tracked file, which itself shows up in the diff).
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

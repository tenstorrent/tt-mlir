# mark_as_system_includes(<target> ...)
# Copies INTERFACE_INCLUDE_DIRECTORIES -> INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
# on each target, so warnings from those headers are suppressed.
function(mark_as_system_includes)
    foreach(_tgt ${ARGN})
        if(TARGET ${_tgt})
            get_target_property(_inc ${_tgt} INTERFACE_INCLUDE_DIRECTORIES)
            if(_inc)
                set_property(TARGET ${_tgt} APPEND PROPERTY
                    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_inc}")
            endif()
        endif()
    endforeach()
endfunction()

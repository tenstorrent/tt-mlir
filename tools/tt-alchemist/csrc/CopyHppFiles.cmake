# CopyHppFiles.cmake
# Generic script to copy only .hpp files from a source directory to a destination directory
#
# Required arguments:
#   -DSOURCE_DIR=<path>     - Source directory to search for .hpp files
#   -DDEST_DIR=<path>       - Destination directory to copy files to

# Validate required arguments
if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

if(NOT DEFINED DEST_DIR)
    message(FATAL_ERROR "DEST_DIR is required")
endif()

# Get all .hpp files recursively from the source directory
file(GLOB_RECURSE HPP_FILES "${SOURCE_DIR}/*.hpp")

list(LENGTH HPP_FILES HPP_COUNT)
message(STATUS "Copying .hpp files from ${SOURCE_DIR} to ${DEST_DIR}")
message(STATUS "Found ${HPP_COUNT} .hpp files")

# Copy each .hpp file maintaining directory structure
foreach(HPP_FILE ${HPP_FILES})
    # Get the relative path from the source directory
    file(RELATIVE_PATH REL_PATH "${SOURCE_DIR}" "${HPP_FILE}")
    
    # Get the destination path
    set(DEST_PATH "${DEST_DIR}/${REL_PATH}")
    
    # Get the destination directory
    get_filename_component(DEST_DIR_FOR_FILE "${DEST_PATH}" DIRECTORY)
    
    # Create the destination directory if it doesn't exist
    file(MAKE_DIRECTORY "${DEST_DIR_FOR_FILE}")
    
    # Copy the file
    file(COPY "${HPP_FILE}" DESTINATION "${DEST_DIR_FOR_FILE}")
endforeach()

message(STATUS "Successfully copied ${HPP_COUNT} .hpp files")
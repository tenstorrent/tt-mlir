# CopyCppFiles.cmake
# Generic script to copy only .cpp files from a source directory to a destination directory
#
# Required arguments:
#   -DSOURCE_DIR=<path>     - Source directory to search for .cpp files
#   -DDEST_DIR=<path>       - Destination directory to copy files to

# Validate required arguments
if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

if(NOT DEFINED DEST_DIR)
    message(FATAL_ERROR "DEST_DIR is required")
endif()

# Get all .cpp files recursively from the source directory
file(GLOB_RECURSE CPP_FILES "${SOURCE_DIR}/*.cpp")

list(LENGTH CPP_FILES CPP_COUNT)
message(STATUS "Copying .cpp files from ${SOURCE_DIR} to ${DEST_DIR}")
message(STATUS "Found ${CPP_COUNT} .cpp files")

# Copy each .cpp file maintaining directory structure
foreach(CPP_FILE ${CPP_FILES})
    # Get the relative path from the source directory
    file(RELATIVE_PATH REL_PATH "${SOURCE_DIR}" "${CPP_FILE}")

    # Get the destination path
    set(DEST_PATH "${DEST_DIR}/${REL_PATH}")

    # Get the destination directory
    get_filename_component(DEST_DIR_FOR_FILE "${DEST_PATH}" DIRECTORY)

    # Create the destination directory if it doesn't exist
    file(MAKE_DIRECTORY "${DEST_DIR_FOR_FILE}")

    # Copy the file
    file(COPY "${CPP_FILE}" DESTINATION "${DEST_DIR_FOR_FILE}")
endforeach()

message(STATUS "Successfully copied ${CPP_COUNT} .cpp files")
# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /localdev/vwells/ttmlir-toolchain-x86/venv/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /localdev/vwells/ttmlir-toolchain-x86/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vwells/sources/tt-mlir

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vwells/sources/tt-mlir

# Utility rule file for install-obj.TTMLIRCAPI.

# Include any custom commands dependencies for this target.
include lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/progress.make

lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI:
	cd /home/vwells/sources/tt-mlir/lib/CAPI && /localdev/vwells/ttmlir-toolchain-x86/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -DCMAKE_INSTALL_COMPONENT="obj.TTMLIRCAPI" -P /home/vwells/sources/tt-mlir/cmake_install.cmake

lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/codegen:
.PHONY : lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/codegen

install-obj.TTMLIRCAPI: lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI
install-obj.TTMLIRCAPI: lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/build.make
.PHONY : install-obj.TTMLIRCAPI

# Rule to build all files generated by this target.
lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/build: install-obj.TTMLIRCAPI
.PHONY : lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/build

lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/CAPI && $(CMAKE_COMMAND) -P CMakeFiles/install-obj.TTMLIRCAPI.dir/cmake_clean.cmake
.PHONY : lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/clean

lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/CAPI /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/CAPI /home/vwells/sources/tt-mlir/lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/CAPI/CMakeFiles/install-obj.TTMLIRCAPI.dir/depend


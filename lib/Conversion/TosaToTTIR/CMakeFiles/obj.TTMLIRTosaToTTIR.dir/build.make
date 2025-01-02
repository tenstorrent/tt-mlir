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

# Include any dependencies generated for this target.
include lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/progress.make

# Include the compile flags for this target's objects.
include lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/flags.make

lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/codegen:
.PHONY : lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/codegen

lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.o: lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/flags.make
lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.o: lib/Conversion/TosaToTTIR/TosaToTTIR.cpp
lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.o: lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.o"
	cd /home/vwells/sources/tt-mlir/lib/Conversion/TosaToTTIR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.o -MF CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.o.d -o CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.o -c /home/vwells/sources/tt-mlir/lib/Conversion/TosaToTTIR/TosaToTTIR.cpp

lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.i"
	cd /home/vwells/sources/tt-mlir/lib/Conversion/TosaToTTIR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vwells/sources/tt-mlir/lib/Conversion/TosaToTTIR/TosaToTTIR.cpp > CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.i

lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.s"
	cd /home/vwells/sources/tt-mlir/lib/Conversion/TosaToTTIR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vwells/sources/tt-mlir/lib/Conversion/TosaToTTIR/TosaToTTIR.cpp -o CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.s

obj.TTMLIRTosaToTTIR: lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/TosaToTTIR.cpp.o
obj.TTMLIRTosaToTTIR: lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/build.make
.PHONY : obj.TTMLIRTosaToTTIR

# Rule to build all files generated by this target.
lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/build: obj.TTMLIRTosaToTTIR
.PHONY : lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/build

lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/Conversion/TosaToTTIR && $(CMAKE_COMMAND) -P CMakeFiles/obj.TTMLIRTosaToTTIR.dir/cmake_clean.cmake
.PHONY : lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/clean

lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Conversion/TosaToTTIR /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Conversion/TosaToTTIR /home/vwells/sources/tt-mlir/lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/Conversion/TosaToTTIR/CMakeFiles/obj.TTMLIRTosaToTTIR.dir/depend

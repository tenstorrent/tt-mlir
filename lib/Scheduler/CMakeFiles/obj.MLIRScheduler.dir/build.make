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
include lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/progress.make

# Include the compile flags for this target's objects.
include lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/flags.make

lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/codegen:
.PHONY : lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/codegen

lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.o: lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/flags.make
lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.o: lib/Scheduler/Scheduler.cpp
lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.o: lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.o"
	cd /home/vwells/sources/tt-mlir/lib/Scheduler && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.o -MF CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.o.d -o CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.o -c /home/vwells/sources/tt-mlir/lib/Scheduler/Scheduler.cpp

lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.i"
	cd /home/vwells/sources/tt-mlir/lib/Scheduler && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vwells/sources/tt-mlir/lib/Scheduler/Scheduler.cpp > CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.i

lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.s"
	cd /home/vwells/sources/tt-mlir/lib/Scheduler && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vwells/sources/tt-mlir/lib/Scheduler/Scheduler.cpp -o CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.s

obj.MLIRScheduler: lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/Scheduler.cpp.o
obj.MLIRScheduler: lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/build.make
.PHONY : obj.MLIRScheduler

# Rule to build all files generated by this target.
lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/build: obj.MLIRScheduler
.PHONY : lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/build

lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/Scheduler && $(CMAKE_COMMAND) -P CMakeFiles/obj.MLIRScheduler.dir/cmake_clean.cmake
.PHONY : lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/clean

lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Scheduler /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Scheduler /home/vwells/sources/tt-mlir/lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/Scheduler/CMakeFiles/obj.MLIRScheduler.dir/depend


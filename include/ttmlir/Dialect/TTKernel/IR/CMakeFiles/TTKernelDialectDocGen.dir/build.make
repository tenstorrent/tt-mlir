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

# Utility rule file for TTKernelDialectDocGen.

# Include any custom commands dependencies for this target.
include include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/compiler_depend.make

# Include the progress variables for this target.
include include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/progress.make

include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen: docs/src/autogen/md/Dialect/TTKernelDialect.md

docs/src/autogen/md/Dialect/TTKernelDialect.md: include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../../../../../docs/src/autogen/md/Dialect/TTKernelDialect.md"
	cd /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTKernel/IR && /localdev/vwells/ttmlir-toolchain-x86/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E copy /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md /home/vwells/sources/tt-mlir/docs/src/autogen/md/Dialect/TTKernelDialect.md

include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/bin/mlir-tblgen
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: include/ttmlir/Dialect/TTKernel/IR/TTKernelAttrInterfaces.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: include/ttmlir/Dialect/TTKernel/IR/TTKernelBase.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: include/ttmlir/Dialect/TTKernel/IR/TTKernelOpsEnums.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: include/ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/CodeGen/SDNodeProperties.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/CodeGen/ValueTypes.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Frontend/Directive/DirectiveBase.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Frontend/OpenACC/ACC.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Frontend/OpenMP/OMP.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/Attributes.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/Intrinsics.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsAArch64.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsAMDGPU.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsARM.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsBPF.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsDirectX.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsHexagon.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsHexagonDep.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsLoongArch.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsMips.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsNVVM.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsPowerPC.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsRISCV.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsRISCVXCV.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsRISCVXTHead.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsRISCVXsf.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsSPIRV.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsSystemZ.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsVE.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsVEVL.gen.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsWebAssembly.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsX86.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsXCore.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Option/OptParser.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/TableGen/Automaton.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/TableGen/SearchableTable.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/GenericOpcodes.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/GlobalISel/Combine.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/GlobalISel/RegisterBank.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/GlobalISel/Target.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/Target.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetCallingConv.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetInstrPredicate.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetItinerary.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetMacroFusion.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetPfmCounters.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetSchedule.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetSelectionDAG.td
include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md: include/ttmlir/Dialect/TTKernel/IR/TTKernelBase.td
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building TTKernelDialect.md..."
	cd /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTKernel/IR && /localdev/vwells/ttmlir-toolchain-x86/bin/mlir-tblgen -gen-dialect-doc -allow-hugo-specific-features -I /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTKernel/IR -I/localdev/vwells/ttmlir-toolchain-x86/include -I/localdev/vwells/ttmlir-toolchain-x86/include -I/home/vwells/sources/tt-mlir/include -I/home/vwells/sources/tt-mlir/include /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelBase.td --write-if-changed -o /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md

include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/codegen:
.PHONY : include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/codegen

TTKernelDialectDocGen: docs/src/autogen/md/Dialect/TTKernelDialect.md
TTKernelDialectDocGen: include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen
TTKernelDialectDocGen: include/ttmlir/Dialect/TTKernel/IR/TTKernelDialect.md
TTKernelDialectDocGen: include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/build.make
.PHONY : TTKernelDialectDocGen

# Rule to build all files generated by this target.
include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/build: TTKernelDialectDocGen
.PHONY : include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/build

include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/clean:
	cd /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTKernel/IR && $(CMAKE_COMMAND) -P CMakeFiles/TTKernelDialectDocGen.dir/cmake_clean.cmake
.PHONY : include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/clean

include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTKernel/IR /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTKernel/IR /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : include/ttmlir/Dialect/TTKernel/IR/CMakeFiles/TTKernelDialectDocGen.dir/depend


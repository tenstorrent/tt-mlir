# Python Bindings
This page aims to clarify, document, and de-mystify the `tt-mlir` python bindings. It will do so by first highlighting the mechanism with which these bindings are generated and exposed to users. It will then document the nuances of `nanobind`, and the different parts of these bindings that must be written in by hand. Finally, it will go through a hands-on example of how to add your own functionality to the `tt-mlir` python bindings.

## `nanobind`
Nanobind is the successor of the ubiquitous `pybind` project. In almost the same syntactical form, it provides a framework to define InterOp between C++ and Python. For more information about `nanobind` specifically, I'd recommend reading through the [documentation](https://nanobind.readthedocs.io/en/latest/index.html). `MLIR` (and by extension: `tt-mlir`) leverages `nanobind` to create bindings for the C++ framework of Dialects, Ops, Types, Attributes, and Passes to be used in Python.

## MLIR in Python
This section highlights the machinery and configuration with which MLIR can be exposed to Python, while still maintaining functional interop with the C++ code. For more context and information feel free to read the [MLIR Python Documentation](https://mlir.llvm.org/docs/Bindings/Python/).

### C-API
While the documentation provides a very lack-lustre explanation as to why the C-API exists, I am here to provide my take on the existence and purpose of the MLIR CAPI.

#### RTTI
MLIR, being a part of the `llvm-project`, follows their ["custom" RTTI](https://llvm.org/docs/ProgrammersManual.html#isa). For this reason, the entire C++ portion of the project isn't built with RTTI to enable to custom functionality. `nanobind`, however, requires RTTI to perform a lot of the casting and transformation required to interop with Python. This conflict leads to the natural desire for an alternative.

C doesn't have RTTI, it's a stable language without the extra convenience and machinery presented in C++. If a C-API were present, the python bindings can link against the C-API, relying on externally defined `NanobindAdaptors` to do the type conversions using `nanobind` mechanisms instead of relying on the C++/LLVM RTTI for the Python bindings.

#### C++ ABI
The C++ Application Boundary Interface (ABI) proves to be a challenging barrier to accessing functionality from C++. Without a _defined_ stable ABI, it becomes difficult to deal with some of the complexity required to package and InterOp with Python. Specifically, dealing with templates, inheritance, and RTTI can prove quite the challenge.

To simplify this process, C provides a relatively stable ABI. The C-API also acts as a wrapper around the complex C++ functions, providing a simple "trampoline" for Python to link against.

### `nanobind` x C-API Functionality
In the previous section, I mentioned `NanobindAdaptors`. This file helps to define some of the key design decisions made when linking the Python bindings against the C-API instead of the underlying C++ API. Functionally, the Python bindings act as a "wrapper" around the CAPI, exposing the functionality through python.

#### `mlir-c/Bindings/Python/Interop.h`
This file is key to defining the

#### `mlir/Bindings/Python/NanobindAdaptors.h`

### Defining the C-API.
For primitives to be defined for use in Python, they must first be implemented in C++. This is outside of the scope of the Python specific code, please refer to the rest of `tt-mlir` documentation for references on this. Once the C++ functionality is defined, the C-API must be constructed on top of this to serve as the layer between

## Generating Bindings
This section will outline the mechanism with which bindings are generated, and the intricacies of this step.



### Declaring Python Bindings
The first step to kicking off binding generation is to declare that they should exist for some dialect. `MLIR` provides a CMake module ([`AddMLIRPython`](https://github.com/llvm/llvm-project/blob/main/mlir/cmake/modules/AddMLIRPython.cmake)) which exposes the following utility functions which can be declared to state what Python bindings are generated. For more information about the specific arguments and expected structure of these CMake functions refer to the `AddMLIRPython` module and `python/CMakeLists.txt`.

#### `declare_mlir_python_sources`
**Overview** \
This function provides an interface to directly copy `.py` source files into the final built python module.

**Key Arguments**
- `ADD_TO_PARENT` defines the Parent `name` to which this source will be added to, inheriting the location.

**Usecases**
- We use it to declare generic "Parents" which contain the generated/declared python files from many of the submodules within the dialects.
- We use it to directly copy over key test infrastructure like `ttir_builder` as purely python programmed modules.

#### `declare_mlir_dialect_python_bindings`
**Overview** \
This function is the key to invoking the mechanism to _generate_ python bindings from Tablegen Definitions.

**Key Arguments**
- `TD_FILE` Relative to `ROOT_DIR`, where the Tablegen Definition file to build bindings off of is located. Note: This currently just forwards the TD files from `include/ttmlir/Dialect`.
- `SOURCES` Raw python files associated with bindings. Note: These files will essentially forward the generated modules forward.
- `GEN_ENUM_BINDINGS_TD_FILE` if `GEN_ENUM_BINDINGS` is `ON`, this will build enum bindings from the defined Tablegen file.
 - `DIALECT_NAME` What name the dialects should be generated under.

**Usecases**
- We use this CMake function to define and generate the bindings for the `ttkernel`, `ttir`, `tt`, and `ttnn` dialects.

#### `declare_mlir_python_extension`
**Overview** \
This is the CMake function used to link C++ Source Files + declared `nanobind`s into the generated python module.

**Key Arguments**
- `EMBED_CAPI_LINKS_LIBS` This is to declare the libraries used to link against the CAPI in the bindings. Learn more in the CAPI section below.
- `PRIVATE_LINK_LIBS` Declares other libraries that are linked against the Python bindings.

**Usecases**
- We use this function to build and link all of our custom `nanobind`s and hand-written Type/Attr bindings into the `ttmlir` module.

#### `add_mlir_python_common_capi_library`
**Overview** \
This function adds a shared library embedding all of the core CAPI libs needed to link against extensions.

#### `add_mlir_python_modules`
**Overview** \
This is the final packaging function of the python bindings, linking all of the sources together and packaging it into a built module.

### Building MLIR Primitives from Tablegen
The `declare_mlir_dialect_python_bindings` leverages a mechanism of the `mlir-tblgen` to build the python bindings for some defined dialect. What are the intricacies of this functionality?

#### [`mlir-tblgen`](https://llvm.org/docs/CommandGuide/mlir-tblgen.html)
This tool parses `.td` Tablegen files to automatically generate C++ code to implement that functionality in MLIR. We leverage the Tablegen language to define our dialects in `tt-mlir`, and this tool is exactly what gets invoked to build and generate the code to functionally use this dialect in our codebase.

#### Trivial Constructors

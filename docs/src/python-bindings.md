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

#### `include/mlir-c/Bindings/Python/Interop.h`
This file is key to defining the InterOp between the C-API and Python w.r.t. maintaining and accessing information in a pointer. It exposes an AI that interfaces immediate data pointers with python `capsule`s. `PyCapsule`s are essentially thin wrappers around data pointers in Python. The critically contain data (`void*`), destructor method, and a name.

Within the Interop, the assumption is that the data's ownership and lifetime is managed by some _bound_ object that was created in C++. This file merely provides the API with which the underlying data pointer is passed around as either a PyCapsule or the raw pointer, and this file provides the type conversion utilities to convert between Python and C from an underlying object.

#### `include/mlir/CAPI/Wrap.h`
This header defines the API to InterOp between C-API objects and their C++ equivalent. By calling `wrap()` on a C++ MLIR object to have the underlying data create a C-API object on the same memory, and `unwrap()` does it the other way around.

They key caveat with this wrapping/unwrapping is the ownership over the lifetime of the data itself. The constructors for almost all of the primitives have already been defined in C++. As such the syntax for creating a new C-API object is more the syntax of creating an object in C++ and wrapping it into a CAPI object. The lifetime of the pointer is therefore maintained by the CAPI object as it gets passed around in return objects.

#### `include/mlir/Bindings/Python/NanobindAdaptors.h`
As the CAPI object gets bounced around in memory, the ownership and lifetime of the data must eventually reach python to be controlled by the user. The implementation details are not relevant to this component as to how the data reaches python. This component provides the utility to create copies of the underlying data and send them through `nanobind`, effectively framing itself as the InterOp component between CAPI objects and their `nanobind` equivalents.

Through the carefully created contract between these components of the MLIR project, the IR primitives are exposed to Python, created in C++, and bounced off of the C-API. While I may have gleaned over the other supporting mechanisms in this explanation, explore the parent directories for these three files for a more detailed look into the semantics of ownership and such.

### Defining the C-API.
For primitives to be defined for use in Python, they must first be implemented in C++. This is outside of the scope of the Python specific code, please refer to the rest of `tt-mlir` documentation for references on this. Once the C++ functionality is defined, the C-API must be constructed on top of this to serve as the "InterOp" layer.

### `get` & Constructing C-API Objects
Since most constructors for IR primitives are created in C++, the goal is to construct objects in C++, but have the ownership exposed to Python. We do this through the creation of a `Get` function. The get function will essentially intake primitive C-types, and invoke the `::get` operator in C++ to construct the object. A simple code example for the `ttkernel.TileType` is shown below:

`include/ttmlir-c/TTTypes.h`
```c

// We export the function outside of the scope of "C" such that it can be defined later using C++ methods.

MLIR_CAPI_EXPORTED MlirType ttmlirTTTileTypeGet(MlirContext get, unsigned height, unsigned width, uin32_t dataType);
```

`lib/CAPI/TTTypes.cpp`
```c++

MlirType ttmlirTTTileTypeGet(MlirContext ctx, unsigned height, unsigned width, uint32_t dataType) {
    // We return the **wrapped** created C++ object, transferring Ownership to the C-API
    return wrap(
        TileType::get(
            unwrap(ctx), // Now we unwrap the MlirContext object to cast it to a mlir::MLIRContext object (w/o affecting ownership)
            llvm::SmallVector<std::int64_t>{height, width}, // We construct the list here since a list isn't natively defined in the C-API,
            static_cast<ttcore::DataType>(dataType) // Here we cast the int value to get the Enum value from `ttcore::DataType`
        ) // Invoking the builtin get operator to create and get the pointer for some object
    );
}
```

The key details to note are the reliance on C++ methods in the `get` definition like initializer lists. By leveraging the InterOp the get method will return a pointer which can easily be represented in the C-API and owned as such, while masking the complexities of the C++ underneath from `nanobind`. Definitions such as these must either be written by hand (as shown above), or they can automatically be generated for certain IR primitives. We will learn more about that below.

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
To deal with automatically generating the functionality around an Operation, a certain amount of _generality_ is needed to deem the problem trivial enough to generate. All of the IR primitives are thankfully able to be constructed from `.td` to their relevant C++ implementations. _However_, as shown in the `TileType` example, the conversion from simple C primitives (+ pre-defined MLIR C-API types) to C++ `get` functions isn't _trivial_. For this reason, we can start to analyze the IR primitives and deem which ones are trivial for C-API generation, and which must be implemented by hand.

- `enum`
    - The enum type can be considered _very_ generic. With the underlying data storage type being integral values, and an optional `String` representation in MLIR. By iterating over all of the user defined enum values, a very trivial constructor can be made to automatically generate `enum`s.
- `operation`
    - Operations are a unique case where the constructor isn't often generic enough; however, the `OperationState` exists as a strictly defined struct which contains all of the relevant construction details and implementation requirements for an operation. For this reason, while it is not trivial, it is _generic_ enough that the `OperationState` can be relied on to form a mechanism which automatically generates C-API builders.
- `Types/Attributes`
    - Types and Attributes unfortunately receive the short end of the stick. Their constructors are wildly generic, and there is no baseline for what is required in the construction of a Type/Attr. For this reason, _at the current moment_ these primitives aren't supported for automatic generation in `mlir-tblgen`, and must be defined by hand.

## Writing Bindings
With the understanding that not all bindings can be automatically generated for us, we can head into the intricacies of defining your own bindings for Types/Attrs.

### LLVM-Style Pythonic "Type Information" + Casting
An important caveat to introduce before entering the domain of writing our own bindings is the understanding of how MLIR approaches the problem of `downcasting` w.r.t. IR primitives. Considering the C-API doesn't have an inheritance structure, Python is required to uphold the inheritance structure and hold the type information such that casting is possible between primitives and their specific implementation (ex: going from MlirAttribute -> TTNNLayoutAttr).

This mechanism can be exposed to Python in multiple different ways, where MLIR supports a specific implementation of an `mlir_attribute_class` and `mlir_type_class` which intake 2 additional C-API functions. To initialize a class using this structure the following functions are required:

- `myAttributeGet`: to construct the Type/Attr
- `myAttributeGetTypeID`: provides a unique static TypeID for `myAttribute`
- `isAMyAttribute`: boolean to see if higher level type is of the same type.

This will then provide an interface where in python a type can be cast by calling the constructor method of some downcasted type:

```py
# Example to downcast using MLIR provided methods.
my_attribute = mlir.MyAttribute(attr: _mlir.ir.MlirAttribute)
```

### Choosing a direct C++ structure instead of C-API
Those who are familiar with the `tt-mlir` python bindings may be aware that our code structure looks drastically different from this, why is that? The answer lies in the redundancy and lack of _extensive_ use of the `nanobind` mechanisms around `tt-mlir` Python bindings.

As mentioned in the C-API section, the C-API is required to form the contract between C++ -> Python, to reduce the collisions with RTTI and the unstable ABI from C++. That being said, it's not _unsupported_ to still directly access C++ members from `nanobind` and skip the C-API Builder functions, instead just opting to create in C++ directly and then `wrap` that result. This is the approach taken "consciously" in the `tt-mlir` python bindings.

What are the consequences of this design decision? The advantages?

#### Direct MLIR Casting Support
Instead of relying on Python for casting, and defining C-API functions to support this functionality; this approach allows us to directly use `mlir::isa`, `mlir::cast`, etc... in it's place.

For example, we support `tt_attribute_class` and `tt_type_class`, which leverage `isa` and `dyn_cast` to downcast to Types and Attrs by wrapping the Python types and operating on the underlying C++ types.

This also brings about some potential collisions with RTTI from `nanobind`. None are present in the bindings (as far as I know), but the bindings are exposed to this problem moving forward.

#### Simpler Initialization Structures
Instead of having to invoke a C-API function to define the `get` method in `nanobind` we can directly invoke the `wrap(CppType::get(...))` functionality that the C-API ends up calling. The primary difference is the native support for complex data structures like `vector` and `map` through `nanobind`. Take for example the following initialization for an attribute:

```c++
// C-API Definition for myAttributeGet
MlirAttribute myAttributeGet(MlirContext ctx, int* array, size_t arraySize) {
    return wrap(MyAttribute::get(ctx, std::vector<int>{array, array + arraySize}));
}

// nanobind direct invocation
tt_attribute_class(m, "MyAttribute")
    .def_static("get", [](MlirContext ctx, std::vector<int> arr) {
        return wrap(MyAttribure::get(ctx, arr));
    })

// nanobind invocation through C-API
mlir_attribute_class(m, "MyAttribute", myAttributeGetTypeId, isAMyAttribute)
    .def_static("get", [](MlirContext ctx, std::vector<int> arr) {
        return myAttributeGet(ctx, arr.data(), arr.size());
    })
// Note: While this may seem like a trivial change, the cost for retaining the function signature in C begins to grow very quickly. Especially when considering maps and more complex data structures.
```

Again, this does come with some nuances w.r.t. the ABI, but for our simple usecase of the bindings it can be considered acceptable...

#### Wait... why are we still defining the CAPI Builders Then?
This leads to an underlying question: What's the point of still defining the CAPI functions if we actually never end up using them? The answer is that we would ideally still maintain the infrastructure to backtrack our changes _if_ we end up making more extensive use of the Python bindings and come across nasty ABI/RTTI issues, or MLIR upstreams significant changes to the Python bindings where we would have to leverage their architecture. With regards to the latter, I have asked some of the contributors and received "iffy" responses, with the general answer being that major changes are _not planned_ for the MLIR Python bindings infrastructure.

That being said, for the low low cost of a few redundant functions being defined, we have a clear backup route in case the Python bindings blow up in our faces. I do think this argument is built on significant personal opinion, in the future we may change the strategy for the bindings. For now, it makes the structure of our python code cleaner, while having a clear route forward if something breaks.

Each MLIR project I've used as a reference approaches the problems differently. AFAIK the bindings are generally defined however the end user desires to invoke them :)

### General Structure
Considering that `mlir-tblgen` will handle the generation of the underlying C++ code, we only need to define the C Builders and the `nanobind`s for each of the Types/Attrs we would like to add.

This often comprises of the following contributions:
- Declaring the C-API Header Function(s) in `include/ttmlir-c`
- Defining the C-API Function(s) in `lib/CAPI`
- Writing out the `nanobind` for that Type/Attr in `python/`.

## Example: Defining `ttkernel` Python Bindings
In this section, we will go through a worked example on the different steps required to expose functionality for the `TTKernel` dialect.

1. We will continue while assuming that the TTKernel dialect has been defined using Tablegen and already has a valid target that compiles the C++ functionality. We will also assume that the current CMake build targets and functionality that uphold the rest of the `ttmlir` dialects already exists.
2. Declare and register the TTKernel dialect in the C-API by calling the `MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TTKernel, ttkernel);` macro in `include/ttmlir-c/Dialects.h`:

```c++
// File: include/ttmlir-c/Dialects.h

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TTKernel, ttkernel);

#ifdef __cplusplus
}
#endif
```
3. Declare CAPI Builder for all of the Types (namely only `CBType` needs to be implemented) in `include/ttmlir-c/TTKernelTypes.h`

```c++
// File: include/ttmlir-c/TTKernelTypes.h

#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirType ttmlirTTKernelCBTypeGet(
    MlirContext ctx, uint64_t port, uint64_t address,
    MlirType memrefType);

#ifdef __cplusplus
}
#endif
```
4. Declare the CAPI builder target in `lib/CAPI/CMakeLists.txt` by adding `TTKernelTypes.cpp` as a source to TTMLIRCAPI.
5. Define the Dialect by formally applying the generated Dialect type into the `CAPI_DIALECT_REGISTRATION` macro.
```c++
// File: lib/CAPI/Dialects.cpp

#include "ttmlir-c/Dialects.h"

#include "mlir/CAPI/Registration.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    TTKernel, ttkernel, mlir::tt::ttkernel::TTKernelDialect)
```
6. Define the CAPI `get` method for `CBType`
```c++
// File: lib/CAPI/TTKernelTypes.cpp

#include "ttmlir-c/TTKernelTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

using namespace mlir::tt::ttkernel;

MlirType ttmlirTTKernelCBTypeGet(MlirContext ctx, MlirType memrefType) {
  return wrap(CBType::get(unwrap(ctx), mlir::cast<mlir::MemRefType>(unwrap(memrefType))));
}
```
7. Define the `nanobind` build target in `python/CMakeLists.txt` by adding `ttkernel` as a dialect, and providing `TTkernelModule.cpp` as a source for `TTMLIRPythonExtensions.Main`.
```cmake
# Define ttkernel dialect
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTKernelBinding.td
  SOURCES dialects/ttkernel.py
  DIALECT_NAME ttkernel
)
```
8. Create `python/dialects/TTKernelBindings.td` to forward the tablegen for TTKernel to the CMake dialect target:
```
include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.td"
```
9. Create `nanobind` module for TTKernel Dialect in `python/TTMLIRModule.cpp`
```c++
// Representation of the Delta you have to add to TTMLIRModule.cpp in the correct locations
NB_MODULE(_ttmlir, m) {
  m.doc() = "ttmlir main python extension";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle ttkernel_handle mlirGetDialectHandle__ttkernel__();
        mlirDialectHandleRegisterDialect(ttkernel_handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(ttkernel_handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  auto ttkernel_ir = m.def_submodule("ttkernel_ir", "TTKernel IR Bindings");
  mlir::ttmlir::python::populateTTKernelModule(ttkernel_ir);
}
```
10. Define `populateTTKernelModule` in `python/TTKernelModule.cpp`
```c++
// File: python/TTKernelModule.cpp
#include <vector>

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/IR.h"
#include "ttmlir-c/TTKernelTypes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

namespace mlir::ttmlir::python {
void populateTTKernelModule(py::module &m) {
  tt_type_class<tt::ttkernel::CBType>(m, "CBType")
      .def_static("get",
                  [](MlirContext ctx, uint64_t port, uint64_t address,
                     MlirType memrefType) {
                    return ttmlirTTKernelCBTypeGet(ctx, port, address,
                                                   memrefType);
                    // Note that for more complex constructors / out of ease this could also be defined using the wrap(CBType::get) style constructor.
                  })
      .def_prop_ro("shape", [](tt::ttkernel::CBType &cb) {
            cb.getShape().vec();
        })
      .def_prop_ro("memref", &tt::ttkernel::CBType::getMemref);
}
} // namespace mlir::ttmlir::python
```
11. Finally, expose the built python bindings using a "trampoline" python file in `python/dialects/ttkernel.py`
```py
from ._ttkernel_ops_gen import *
from .._mlir_libs._ttmlir import register_dialect, ttkernel_ir as ir

# Import nanobind defined targets into ttkernel.ir, and the rest of the generated Ops into the top-level ttkernel python module.
```

### Concluding The Example
While there are quite a few steps for adding a whole new dialect, often times more than not you will only need a subset of these steps to add a new Type/Attr to some existing dialect. Even less to modify the signature of some existing Type/Attr in the bindings.

## Using the Python Bindings
This section will cover the basics of using the Python bindings. I think the folks at MLIR have produced [documentation](https://mlir.llvm.org/docs/Bindings/Python/) that can help you get up to speed quickly. This section will go over some of the nuances of using the python bindings that `ttmlir` has defined explicitly.

### Interfacing with Generated Op Classes
The unfortunate reality is that documentation for autogenerated Ops isn't present. Fortunately, argument names are preserved and the function structure can be invoked by leveraging the `help` function in python. Iteratively running through the functions you want to implement can be helpful.

### MLIRModuleLogger
Almost all of the python bindings behave exactly as expected coming from the `ttmlir` python bindings. A weird addition I think would provide some more context on `nanobind` and managed memory would be the `MLIRModuleLogger`.

This class is defined in C++ to attach to an existing `MLIRContext`, adding hooks to save the module to a `std::vector<std::string, std::string>`. Binding this forward through `nanobind` requires some delicacy about the _state_ of this MLIRModuleLogger object. It needs to modify memory managed by C++, but it attaches to a context that exists in Python. This state management is done through `nanobind` owning and managing a thinly wrapped pointer to the C++ object by setting the return_value policy.

Using the Python bindings when traversing frequently through memory outside of the IR primitives requires some delicacy to ensure data is preserved and the code functions as intended.

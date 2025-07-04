# TT-MLIR Coding Guidelines

This document outlines the coding standards used in the tt-mlir project. These guidelines are designed to enhance the readability and maintainability of our shared codebase. While these guidelines are not strict rules for every situation, they are essential for maintaining consistency across the repository.

Our long-term aim is to have the entire codebase adhere to these conventions.

Since our compiler is built on the LLVM MLIR framework, we strive to align closely with the LLVM coding style guidelines outlined here: [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html).

## Naming

Clear and descriptive names are crucial for code readability and preventing bugs. It’s important to choose names that accurately reflect the semantics and purpose of the underlying entities, within reason. Avoid abbreviations unless they are widely recognized. Once you settle on a name, ensure consistent capitalization throughout the codebase to avoid confusion.

The general naming rule is to use camel case for most names (for example, WorkaroundPass, isRankedTensor())

* Type Names
    * Applies to classes, structs, enums, and typedefs.
    * Should be nouns that describe the entity's purpose.
    * Use upper camel case (for example, TTNNOptimizerOptions, DecompositionPass).
* Variable Names
    * Should be nouns, as they represent state.
    * Use lower camel case (for example, inputLayout).
* Function Names
    * Represent actions and should be verb phrases
    * Use lower camel case (for example, createTTNNOptimizer(), emitTTNNAsCpp()).

## Includes

We prefer #includes to be listed in this order:

1. Main Module Header
2. Local/Private Headers
3. LLVM project/subproject headers (clang/..., lldb/..., llvm/..., etc)
4. System #includes

Each category should:
* Be sorted lexicographically by the full path.
* Be separated by a single blank line for clarity.

Only the [standard lib header](https://en.cppreference.com/w/cpp/header) includes should use <> whereas all the others should use quotes "". Additionally, all project headers must use absolute paths (rooted at ttmlir) to prevent preprocessor and namespacing issues. For example, the following is preferred:

```
#include "ttmlir/module/something.h"
```

over:

```
#include "something.h"
```

Using TTIRToTTNN.cpp as an example, this is what includes would look like for us:

``` c++
#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"  # main header

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"  # these are local/private headers
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"  # llvm project/subproj headers
#include "llvm/Support/LogicalResult.h"

#include <cstdio>  # system includes
#include <algorithm>
```

## Comments

Write comments as full sentences, starting with a capital letter and ending with a period. Comments should explain why the code exists, not just what it does. Use comments to clarify logic, assumptions, or any non-obvious aspects of the code.

Example of a comment:

```
// Initialize the buffer to store incoming data from the network.
```

In general, C++ style comments (//) should be used. Use C-style comments (/**/) only for when documenting the significance of constants used as actual parameters in a call:

```
object.callFunction(/*arg0=*/nullptr);
```

Every function, class, or non-trivial piece of logic should have a comment. Avoid redundant comments for self-explanatory code, but never leave complex code unexplained. Example of redundant comment:

``` c++
// Increment the counter by 1.  // Redundant, avoid.
counter++;
```

Ensure comments are accurate and reflect the current state of the code. Outdated or misleading comments can be worse than no comments at all.

## Code Denesting (Inversion)

Strive to minimize unnecessary indentation without compromising code clarity. One effective way to achieve this is by using early exits and the continue keyword in long loops.

Consider following example:

```c++
void doSomething(Operation *op)
{
    if (op->getNumOperands() > 0
        && isDpsOp(op)
        && doSomethingDifferent(op))
    {
        // ... some long code ...
    }
}
```

It is strongly recommended to format the code as follows:

```c++
void doSomething(Operation *op)
{
    // ...
    // We need to do something with the op that has more than 0 operands
    if (op->getNumOperands() <= 0 ) return;

    // We need something to do with the DPS op
    if (!isDpsOp(op)) return;

    // Just for example purposes
    if (!doSomethingDifferent(op)) return;

    // .. some long code ...
}
```

This reduces loop nesting, makes the reasoning behind the conditions clearer, and signals to the reader that there is no subsequent else to worry about, reducing cognitive load. This can significantly improve code readability and comprehension.

## Function Declaration and Definition Order

To improve code readability and maintainability, we should adopt a consistent approach for organizing function declarations and definitions within a file. The goal is to make it easier for readers to follow the logical flow of function dependencies.

Follow a bottom-up call order:
* Arrange functions so that lower-level helper functions are defined first, followed by higher-level functions that call them.
* This allows each function to be defined after its dependencies, making it clear which functions rely on which.
* For example, if function A calls A1 and A2, then the preferred order is:

```c++
void A1();
void A2();
void A(){
  A1();
  A2();
}
```

Group related functions together:
* If functions are only relevant to a specific “parent” function (for example, A1 and A2 are only called by A), place them directly before the “parent” function.
* If a function (like A2) is also called by other functions (for example, B), place it where it fits the overall bottom-up order.

Avoid mixed ordering:
* Mixing top-down and bottom-up call orders within the same file can make the code hard to read and maintain.

Example of a preferred order:

```c++
void A1() {
  /*...*/
}
void A2() {
  /*...*/
}
void B() {
  A2(); // A2 is defined before B, so dependencies are clear.
}
void A() {
  A1();
  A2();
  B();
}
```

## Helper Functions

These coding guidelines address visibility and linkage of simple helper functions to ensure clarity, prevent linking errors, and improve maintainability:

* If a helper function needs to be defined in a .cpp file, it should be declared static or wrapped inside an anonymous namespace.

* If a helper function needs to be defined in a header file (for example, for templated or performance-critical code), it should be marked as inline.

> [!NOTE]
> A significant concern with declaring functions as non-public (for example, static functions
> or functions in unnamed namespaces) is that they cannot be unit tested in isolation.
> This limitation hinders our ability to write focused, granular tests that verify the
> correctness of individual components and it also reduces test coverage.

## Using Namespaces

Namespaces are an important part of C++ programming, providing a way to organize code and avoid naming conflicts. Choose namespace names that reflect the purpose or functionality of the code contained within.

Follow these guidelines when defining namespaces:

* Use lower-case letters for short, single-word names or those with a clear acronym (for example, ttnn, mlir).
* Use nested namespaces to group logically related code, avoiding too deep or unnecessarily complex hierarchy

Follow these guidelines when using namespaces:

* Do not use a using-directive to make all names from a namespace available because it pollutes the namespace.

```c++
// Forbidden -- This pollutes the namespace.
using namespace std;
```

* Avoid placing code in the global namespace to reduce the potential for name conflicts and ambiguity. Always use specific namespaces. If necessary to use something from the global namespace (such as std), use an explicit std:: prefix rather than importing everything using using namespace std;.
* Do not use namespace aliases at namespace scope in header files except in explicitly marked internal-only namespaces, because anything imported into a namespace in a header file becomes part of the public API exported by that file.
* Try to avoid mixing concepts from different namespaces in a single function or class. If a function belongs to one namespace but calls classes from others, ensure the relationships are clear.
* Wrap classes/structs declared in .cpp files inside of an [anonymous namespace](https://en.cppreference.com/w/cpp/language/namespace#Unnamed_namespaces) to avoid violating [ODR](https://en.cppreference.com/w/cpp/language/definition). See [LLVM docs](https://llvm.org/docs/CodingStandards.html#anonymous-namespaces) for more detailed information.

## Using Alternative Tokens (`and`, `or`, `xor`, etc.)

Although they are standard, we should avoid their use. They are very rarely used in practice and the C++ community widely uses the standard operators (&&, ||, !, etc.), as they are more familiar and easily recognizable to most C++ developers. Their usage can make the code harder to read and maintain, especially for developers who are not familiar with these alternatives. We should stick to the standard operators (&&, ||, !, etc.) for clarity, consistency, and compatibility with other C++ developers and tools.

## Type Aliasing

When declaring type aliases in C++ prefer using over typedef. using provides better readability, especially for complex types, and supports alias templates. Here is an example:

```c++
// Preferred
using Callback = void(*)(int, double);

// Avoid
typedef void (*Callback)(int, double);
```

Choose alias names that clarify their role in the code. Avoid overly generic names that might obscure the type’s purpose. Do not create a type alias unless it significantly improves clarity or simplifies complex types.

## Using `auto` to Deduce Type

Use auto only when it enhances code readability or maintainability. Avoid defaulting to “always use auto.” Instead, apply it thoughtfully in the following scenarios:
  * When the type is immediately clear from the initializer, such as in cast<Foo>(...).
  * When the type is obvious from the context, making the code cleaner and more concise.
  * When the type is already abstracted, such as with container typedefs like std::vector<T>::iterator.

In all other cases, prefer explicit type declarations to maintain clarity and ensure the code remains easy to understand.

## Python Coding Guidelines

### Python Version and Source Code Formatting
The current minimum version of Python required is 3.10 or higher. Python code in the tt-mlir repository should only use language features available in this version of Python.

The Python code within the tt-mlir repository should adhere to the formatting guidelines outlined in PEP 8.

For consistency and to limit churn, code should be automatically formatted with the black utility, which is PEP 8 compliant. Use its default rules. For example, avoid specifying --line-length even though it does not default to 80. The default rules can change between major versions of black. In order to avoid unnecessary churn in the formatting rules, we currently use black version 23.x.

When contributing a patch unrelated to formatting, you should format only the Python code that the patch modifies. When contributing a patch specifically for reformatting Python files, use black, which currently only supports formatting entire files.

Here is a quick example, but see the black documentation for details:

```bash
$ black test.py                    # format entire file
```

# Introduction

The following document provides an overview of the TT-MLIR
project, with a focus on the technical specifications of an MLIR-based
compiler stack. So what exactly is an MLIR-based compiler stack? MLIR
(Multi Level Intermediate Representation) is a subproject coming out
of the LLVM Project. It seeks to introduce extensibility and sustainable
code design to a very modular compiler framework. This essentially means
to take a much larger more involved compiler (like LLVM) and split it
into sub-compilers that each produce their own Intermediate
Representation (IR) of what you've fed the compiler.

Disclaimer: This is intended to be a working document, if you find something incorrect or incomplete please feel free to create a PR.

## Motivations

The idea of having a multi-level IR might not seem so far fetched, in
fact it resembles some of our current software
stacks. The idea of going from a High Level TVM Graph → Lowered PyBUDA Graph →
Netlist, with each layer having their own level of optimizations is
quite a familiar concept. However, there are problems with the
reusability and integration of optimizations for the current software compiler
stack. Currently, users are almost forced to choose between a top-down
optimization or bottom-up optimization, with both requiring
"expert-level" expertise to optimize for desired performance. Developing
2 entirely different projects is taxing, and it's hard to translate the
benefits of BUDA over to metal (or the other way around). One of the primary
goals of tt-mlir is to enable a consistent programming model between software
stacks, concepts for improving optimizations in the compiler stack should 1-1
carry over to hand-written TTNN.

The benefits grow even further when one can understand all the possible entry points that
multiple IRs present. Existing MLIR based projects like [OpenXLA](https://openxla.org) and [torch-mlir](https://github.com/llvm/torch-mlir)
can natively output MLIR in a dialect that can be transcribed to the
TTIR dialect as well!

# What is MLIR and why use it?

MLIR is a compiler infrastructure that is designed to be modular and extensible. The main benefits the tt-mlir project hopes to gain by using MLIR include:

- Industry Standard Compiler Framework
  - Lots of boilerplate algorithms, data structures, and useful software that is common to compiler development
- Ecosystem
  - Hook into existing front-end MLIR projects
- Testing framework
  - A battle-tested test infrastructure that will enable us to write fine grained tests and rely less on end-to-end testing
  - Common IR Serialization Format that's easy to test, debug, and edit

Additional documentation to highlight the benefits of MLIR can be found here:
- [MLIR Whitepaper](https://ieeexplore.ieee.org/abstract/document/9370308)
- [MLIR Documentation](https://mlir.llvm.org)

## MLIR: Overview

MLIR is at it's root an interpreter that can parse "readable" text in
some .mlir format. The unique properties lie in the modularity of the
parsing itself. MLIR is built upon a collection of **Dialects**, each of
these Dialects define a collection of **Operations**, **Types**, and
**Attributes**. These dialects follow their own syntax, and they can encode
any amount of information. The benefit is that MLIR provides bindings
and hooks such that a user can directly translate these IRs into usable
artifacts for that layer of complexity. An example of this would be the
relatively high level TOSA Dialect, which is used to represent computation
over tensors, and then lowering that to a more hardware specific dialect
that closely models the programming model of the hardware or underlying backend.
It is the dialect system itself which powers the
multi-level functionality of MLIR, with different dialects a user can
essentially "lower" through their software stack by just transforming
between the different dialects for their layers. Dialects can exist in a
broad range from purely mathematical dialects, to a LinAlg Dialect, or a
Tensorflow Dialect defined for ML Graphs. Each dialect encodes its own
information and their operations can use the Types/Attributes of other
dialects as parameters. Multiple dialects are possible in one module,
and encouraged to highlight optimizations of different dialects. In our
usecase for the TT Stack, MLIR acts a "mid-level" compiler which makes
the task of joining together various entry points and backends much
simpler.

### MLIR Primitives

So what does MLIR look like, how does it work and get parsed? The
hierarchy of an MLIR module is as shown:
```mlir
#permutation = array<i64: 0, 2, 1>

module {
  func.func @forward(%input: tensor<32x64x128xf32>) -> tensor<32x128x64xf32> {
    %output = ttir.empty() : tensor<32x128x64xf32>
    %result = "ttir.permute"(%input, %output) <{permutation = #permutation}> : (tensor<32x64x128xf32>, tensor<32x128x64xf32>) -> tensor<32x128x64xf32>
    return %result : tensor<32x128x64xf32>
  }
}
```

-   Attributes (defined using #)

    -   The syntax of actually creating an attribute is modular, and
        custom assembly instructions for different attributes can be
        applied.

-   Operations

    -   These operations are accessed with the . method, so you'll see
        some examples like `func.func` or `ttir.empty`. Each operation
        also provides it's own assembly instructions but often strictly
        defines the type of result

    -   Quotes are added around `ttir.permute` since it's part of a
        custom dialect.

    -   Operations typically have operands (arguments) and results which
        are highlighted with %, these results and operands help to show
        the relationship between operations

-   Types

    -   Types are shown as dataformats throughout *this* compiled mlir
        module, where tensor and array are some examples.

    -   They help to demonstrate the transformation of information and
        its representation as it's processed across this module.

### MLIR Workflow

The overall MLIR workflow doesn't necessarily involve writing .mlir files
or even modifying them. The Intermediate Representations are truly just
representations, we can parse them to demonstrate what the graph looks
like at that current stage of optimization, or run a **pass** through
them to optimize certain functions. The overall framework is designed
with the following architecture in mind:

1.  Graph Information exists

2.  Graph Information is transformed (through any which method) into a
    high-level MLIR representation

3.  **Passes** are run on the high-level implementation to lower into
    TTIR, a common IR that can be lowered into multiple backends

4.  Depending on the usecase more passes are run to lower to whatever
    backend the user would like (ex: TTNN Backend)

### What are Passes?

Transformations in MLIR are represented as passes that occur during the
parsing of some information. These passes can be executed when parsing
or generating MLIR modules. These transformations can have a myriad of
purposes, and are completely user defined as to how they modify the
module. Some examples of passes can be for lowering purposes as
mentioned before, where a dialect is parsed and then each operation is
transformed to a lowered dialect following some set of user defined
rules. Passes are also used for optimizations and backend code
transformation in the context of this project. They're a powerful tool
and provide most of the functionality to transform between layers of
dialects, and they provide a simple platform for modifications of an
MLIR module.

## Why not make our own?

Now that I've described the functionality of the MLIR framework, it
seems like making an in house multi level Intermediate Representation
system would be pretty similar, so why are we going through the effort
of implementing this framework?

One of the biggest reason can be attributed to the active developer
community surrounding the project, being a part of the LLVM Project
means that there is solid developer support, and the framework is
designed to be a tool for many different paradigms of compute. This
scalability and strong mission statement lend to the strengths of MLIR
being a solid platform to use as a middle layer in our compiler stack.
Furthermore, as a functional benefit of being part of a larger open
source project, MLIR has a whole library of tests and infrastructure
that we can leverage for solid code health while starting a new project.

### Automation

It's not only about developer support, another key benefit of MLIR is
that it's built with autogeneration in mind. Through TableGen a lot of
the boilerplate of creating this multi-level IR become abstracted away
to truly focus on implementation and execution. This automation is built
on top of a pre-existing robust framework with a lot of implementations
and support from other large players in the ML scene. By integrating
with these automation pipelines, we allow for external developers to
have a much simpler entry-point into our software stack!

# TT-MLIR: Bringing MLIR to the TT Stack

Now that we have defined this pretty cool project, let's look at the
implementation details of bringing MLIR (and related optimizations) into
the TT Stack. Since it acts as a mid-level compiler we can start by
defining the "bottom" and "top" layers of the compiler. BUDA already has
a well defined set of frontend optimizations to some TVM defined graph
and is knowledgeable of the hardware that these models want to run on.
We want to interrupt the BUDA stack to only give us the frontend
compiled graph before any hardware specific lowering is to occur. What
this will produce is information that is agnostic to different backends
and their execution on TT hardware, but this is still valid information
to optimize at *different levels* for later compilation. The "bottom" of
our graph is now defined as the backend that will produce the
machine-specific code to be executed. While MLIR could allow for any
level of complexity downwards for the bottom, we will define a very
aggressive TTNN backend for the MVP.
Desired Optimization List:

-   Forge-FE (frontend)

    -   Graph Optimizations, Constant Folding, Operation Fusion

-   TT-MLIR (mid-level)

    -   Data Storage, Memory Configuration, Grid Configuration

-   TTNN (backend)

    -   Kernel Configuration\*, Network Optimization

*\*Subject to Change / Be Moved to TT-MLIR*

## **TT-MLIR Dialects**

Now that we have defined the series of optimizations that we would like
to see implemented in TT-MLIR, we can begin to help define the dialects
that would help to support these different levels of optimizations. For
more detail on each of these dialects, please refer to the GitHub Wiki
and TableGen descriptors. I think that Nick does a great job of
documenting the key functionality.

### TT Dialect
The TT Dialect is **only** for common Types and Attributes used throughout the many levels of the mid level compiler.

### TTIR Dialect

The TTIR Dialect is defined as the common dialect for TT-MLIR, as such it doesn't define anything hardware/backend specific. It lists out general actions that would take place on TT hardware such as dispatch, layout, and kernel operations.

#### Generic Operation

This is one of two operations that's crucial to understand the intended optimization characteristics of the TTIR Dialect. The generic operation dictates the actions that would be taken to dispatch some instruction to TT hardware such that it executes some instruction. Parametrically, the operation consumes inputs, outputs, maps to read the tensors, and access-types to the memory. These parameters highlight the optimizations that can be performed at this level to change the location of the memory, transpose using variant access maps, or even the grid upon which the computation takes place. The operation also contains a block in which the exact behaviour for that operation to occur is stored.

#### Layout Operation

The layout operation is key in describing the storage of memory throughout the execution graph. Layout determines the sharding spec, location of the memory, data types, and tile sizes of some tensor. While generic describes the dispatch for some data-wise transformation to take place, the data itself is laid out across the chip through the layout operation.

Both of these operations describe the key functionality of the TTIR dialect and the optimization space that it provides.

## Built-in MLIR Dialects

The functionality of TT-MLIR Dialects also depends / is inspired by the
functionality of Built-in MLIR Dialects like Affine and LinAlg. Below
are summaries of some of the key members of these Dialects

### Affine Dialect

\[[Reference](https://mlir.llvm.org/docs/Dialects/Affine/#affine-maps)\]
Affine maps help to describe transformations on coordinate systems,
while this may not really make sense, imagine trying to index a rank 2
tensor. By getting t\[x, y\] I can access the element in the Xth row and
Yth column, but if I wanted to transpose the tensor I might have to
re-layout the entire tensor such that the data would be accessible using
t\[x, y\] to get the element in the Yth row and Xth column. This
transpose can also be represented using an Affine Map to transform (x,
y) -\> (y, x) and this would let the tensor data remain in place while
the access method is modified. This extends even further to more complex
transformations such that stride lengths or unique indexing methods can
be implemented without complicated manipulation.

### Tensor Dialect

\[[Reference](https://mlir.llvm.org/docs/Dialects/TensorOps/)\]
The tensor dialect defines the functionality and Type of the fundamental
Tensor. This dialect contains members that would represent manipulation
and representation of tensors as multi-dimensional data with shapes and
datatypes. Not much else is different about this dialect, the reference
covers key topics if implementation details are needed.

### Func Dialect
\[[Reference](https://mlir.llvm.org/docs/Dialects/Func/)\]

### TOSA Dialect
\[[Reference](https://mlir.llvm.org/docs/Dialects/TOSA/)\]

### SCF Dialect
\[[Reference](https://mlir.llvm.org/docs/Dialects/SCFDialect/)\]

### EmitC Dialect
\[[Reference](https://mlir.llvm.org/docs/Dialects/EmitC/)\]

## tt-explorer - Performance Optimization Tool

A unique project related to TT-MLIR is the integration of Performance
Optimization Tools such that users are easily able to visualize and
readily tune their models without needing an expert level understanding
of the tech stack.
['tt-explorer'](./tt-explorer/tt-explorer.md)
is built with Google AI's [Model
Explorer](https://github.com/google-ai-edge/model-explorer)
as a base for the visualization tool, and a [custom
adapter](https://github.com/vprajapati-tt/tt-adapter) to
parse TT-MLIR projects. This would allow users to readily tune their
models, and optimize for the TTIR layer (ex: they can change certain
memory to be laid out in L1 instead of DRAM, or change the grid layout
of an operation to be larger than what was previously assigned). After
compilation with these overrides, the runtime information can then be
fed directly into a Tracy Performance Analysis for the user to visualize
the impacts of their tuning, seeing which operations were least
performant and continuing in a gamified design loop for iterative
performance tuning!

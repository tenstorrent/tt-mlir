.. ttir-builder documentation master file, created byMore actions
   sphinx-quickstart on Wed Jun 18 13:55:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the ttir-builder documentation!
==========================================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.

**ttir-builder** is a tool for creating TTIR operations. It provides support for MLIR modules to be generated from user-constructed ops, lowered into TTNN or TTMetal backends, and finally translated into executable flatbuffers. Or you can do all three at once!

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Contents
--------

.. toctree::
   :maxdepth: 2

   ttir-builder/apis
   ttir-builder/utils
   ttir-builder/ops
   ttir-builder/golden

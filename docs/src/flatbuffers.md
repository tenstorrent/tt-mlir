# Flatbuffers

Flatbuffers are the binary serialization format used by TTMLIR and they
currently come in a few flavors (designated by the file extension):

- `.ttsys`: A system description file that is the mechanism for supplying target
  information to the compiler. These can be collected on a target machine and
  downloaded to a development machine to enable cross-compilation.
- `.ttnn`: A compiled binary file intended to be loaded and executed by the
  TTNN backend runtime.
- `.ttb`: A compiled binary file intended to be loaded and executed by the
  TTMetal backend runtime (Unsupported).

// RUN: ttmlir-translate -mlir-to-python %s | FileCheck %s

// CHECK-LABEL: class Model(torch.nn.Module):
// CHECK: def __init__(self):
// CHECK: self.weight = ttnn.load_weight()
// CHECK: def forward(self, input):
// CHECK: return ttnn.matmul(input, self.weight)
module {
  emitpy.class @Model(#emitpy.opaque<"torch.nn.Module">) {
    func.func @__init__(%self: !emitpy.opaque<"Model">) {
      %0 = emitpy.expression : -> !emitpy.opaque<"ttnn.Tensor"> {
      ^bb0():
        %1 = emitpy.call_opaque "ttnn.load_weight"() : () -> !emitpy.opaque<"ttnn.Tensor">
        emitpy.yield %1 : !emitpy.opaque<"ttnn.Tensor">
      }
      emitpy.set_attr %self, "weight", %0 : (!emitpy.opaque<"Model">, !emitpy.opaque<"ttnn.Tensor">)
      return
    }

    func.func @forward(%self: !emitpy.opaque<"Model">,
                       %input: !emitpy.opaque<"ttnn.Tensor"> {emitpy.name = "input"}) -> !emitpy.opaque<"ttnn.Tensor"> {
      %0 = emitpy.expression(%self, %input) : (!emitpy.opaque<"Model">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
      ^bb0(%s: !emitpy.opaque<"Model">, %i: !emitpy.opaque<"ttnn.Tensor">):
        %w = emitpy.get_attr %s, "weight" : (!emitpy.opaque<"Model">) -> !emitpy.opaque<"ttnn.Tensor">
        %r = emitpy.call_opaque "ttnn.matmul"(%i, %w) : (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor">
        emitpy.yield %r : !emitpy.opaque<"ttnn.Tensor">
      }
      return %0 : !emitpy.opaque<"ttnn.Tensor">
    }
  }
}

// CHECK-LABEL: class WithMethodKinds:
// CHECK: @classmethod
// CHECK: def make(cls):
// CHECK: @staticmethod
// CHECK: def util(x):
module {
  emitpy.class @WithMethodKinds {
    func.func @make(%cls: !emitpy.opaque<"C"> {emitpy.name = "cls"}) attributes {emitpy.method_kind = "classmethod"} {
      return
    }
    func.func @util(%x: !emitpy.opaque<"int"> {emitpy.name = "x"}) attributes {emitpy.method_kind = "staticmethod"} {
      return
    }
  }
}

// CHECK-LABEL: class NoBase:
// CHECK-NEXT: pass
module {
  emitpy.class @NoBase {
  }
}

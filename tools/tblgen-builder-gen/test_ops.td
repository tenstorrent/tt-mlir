// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Simplified TTIR ops for testing the generator

class TTIR_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> {
    let summary = "Base class for elementwise operations with one input tensor.";
}

def TTIR_SigmoidOp: TTIR_ElementwiseUnaryOp<"sigmoid"> {
    let summary = "Eltwise sigmoid.";
    let description = [{
      The `sigmoid` operation computes the sigmoid of each element in the input tensor.

      Mathematical definition: sigmoid(x) = 1 / (1 + exp(-x))
    }];
}

def TTIR_ReluOp: TTIR_ElementwiseUnaryOp<"relu"> {
    let summary = "Eltwise ReLU.";
    let description = [{
      The `relu` operation computes the rectified linear unit of each element.

      Mathematical definition: relu(x) = max(0, x)
    }];
}

def TTIR_CosOp: TTIR_ElementwiseUnaryOp<"cos"> {
    let summary = "Eltwise cosine.";
    let description = [{
      The `cos` operation computes the cosine of each element in the input tensor.

      Mathematical definition: cos(x)
    }];
}

def TTIR_AbsOp: TTIR_ElementwiseUnaryOp<"abs"> {
    let summary = "Elementwise absolute value operation.";
    let description = [{
      The `abs` operation computes the absolute value of each element.

      Mathematical definition: abs(x) = |x|
    }];
}

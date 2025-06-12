#!/usr/bin/env python

import argparse
import sys
import traceback
from typing import List, Set

import mlir

# -----------------------------------------------------------------------------

class node (object):
  def __init__(self, ID : str):
    self.ID = ID
    self.successors = set ()

  @property
  def successors (self) -> Set:
    return self.successors

class OpVisitor(mlir.NodeVisitor):
    def visit_Operation(self, node: mlir.astnodes.Node):
        print(f"OP: {node}\n")


def main (args):
  with open (args.file, 'r') as input:
    ast = mlir.parse_file (input)
    print ('parsed')
    # print (ast.dump ())
    OpVisitor().visit(ast)

# ............................................................................

def _parse_args (argv):
    p = argparse.ArgumentParser ()

    p.add_argument (dest='file', metavar='FILE')

    return p.parse_args (argv)

if __name__ == '__main__':
    args = _parse_args (sys.argv [1:])
    try:
        main (args)
    except Exception as e:
        trace = traceback.format_exc (chain=True)
        print ("------\nexception caught:", getattr (e, 'message', str (e)), '\n------\n', trace)
        sys.exit (1)

# -----------------------------------------------------------------------------

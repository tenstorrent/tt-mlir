#!/usr/bin/env python

import argparse
from pprint import pprint
import sys
import traceback
from typing import List, Set

from ttmlir import ir

# -----------------------------------------------------------------------------

class node (object):
  def __init__(self, ID : str):
    self.ID = ID
    self.successors = set ()

  @property
  def successors (self) -> Set:
    return self.successors


def main (args):
  with ir.Context() as ctx:
     with open (args.file, 'r') as input:
        module = ir.Module.parse(input.read())
        print ("parsed")

        ops = []
        for op in module.body.operations:
          for region in op.regions:
            for block in region.blocks:
              for op in block.operations:
                ops.append(op)
                print (dir(op))
                print (f"{op.result.get_name()} <- {op.operation.get_name()}")
                for operand in op.operands :
                   print (f"\t{operand.get_name()}")
                   print (f"\t{operand.type}")
        print (f"loaded {len(ops)} ops")

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

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Topic kernel bank for gen_knowledge_09_3_D.py.

Each topic maps to a dict with keys:
  _cat: category label
  what: definition kernel
  how:  mechanism kernel
  why:  significance kernel
  vs:   comparison kernel
  ex:   example kernel
  mis:  misconception kernel

The composer in the driver script wraps these kernels with prelude/intro/tail
prose to produce full ShareGPT answers.

Target: ~85 topics x 6 sub_angles = ~510 (topic, angle) seeds. Driver picks
500 deterministically with seed 1009309D.
"""
T = {}

# Pull in kernels from split modules to keep this file readable.
from gen_knowledge_09_3_D_data2 import register as _r2
from gen_knowledge_09_3_D_data3 import register as _r3
from gen_knowledge_09_3_D_data4 import register as _r4

_r2(T)
_r3(T)
_r4(T)

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Topic kernel bank for gen_knowledge_09_3_G.py.

Each topic maps to a dict with keys:
  _cat: category label
  what: definition kernel
  how:  mechanism kernel
  why:  significance kernel
  vs:   comparison kernel
  ex:   example kernel
  mis:  misconception kernel

Target: ~85 topics x 6 sub_angles >= 510 (topic, angle) seeds. Driver
selects 500 deterministically with seed 1009312G.

All topics are fresh vs sibling generators A/B/C/D/E/F.
"""
T = {}

from gen_knowledge_09_3_G_data2 import register as _r2
from gen_knowledge_09_3_G_data3 import register as _r3
from gen_knowledge_09_3_G_data4 import register as _r4
from gen_knowledge_09_3_G_data5 import register as _r5
from gen_knowledge_09_3_G_data6 import register as _r6
from gen_knowledge_09_3_G_data7 import register as _r7

_r2(T)
_r3(T)
_r4(T)
_r5(T)
_r6(T)
_r7(T)

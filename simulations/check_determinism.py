"""Determinism check: run small versions of each simulation engine in two fresh
Python processes and compare the outputs byte for byte.

Usage: python check_determinism.py
"""
import hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

SNIPPET = r'''
import json, sys
sys.path.insert(0, ".")
from sim import run
from dgps import DGPS
from run_example import mse_ratio
import numpy as np
out = {
  "validation": [float(x) for x in np.concatenate(run(DGPS["cexp2_full"], 200, 2000, 1000, seed=12))[:50]],
  "example": mse_ratio(0.8, 2.0, 0.4, 500, 4000, seed=13),
}
print(json.dumps(out, sort_keys=True))
'''


def one():
    r = subprocess.run([sys.executable, "-c", SNIPPET], capture_output=True, text=True,
                       check=True, cwd=HERE)          # the snippet imports sibling modules
    return r.stdout


def main():
    a, b = one(), one()
    ha, hb = hashlib.sha256(a.encode()).hexdigest(), hashlib.sha256(b.encode()).hexdigest()
    print("process 1:", ha)
    print("process 2:", hb)
    assert a == b, "outputs differ between fresh processes"
    print("identical: yes")


if __name__ == "__main__":
    main()

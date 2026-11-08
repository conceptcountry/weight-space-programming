#!/usr/bin/env python3
"""Seed-stability check on the one borderline CI, then emit numbers.typ.

RESULTS2.md reports IFBench ctrl->edit as +4.7 [+0.0, +9.0] ("touches 0").
emit_results.py at seed 20261107 / 10k reps gives [+0.33, +9.33] (excludes 0).
A verdict clause must not depend on a bootstrap seed, so quantify the wobble
across many seeds before letting the paper claim either reading.
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = json.load(open(os.path.join(HERE, "results", "results.json")))


def scores(arm, bench):
    d = json.load(open(os.path.join(DATA, f"g_{arm}_{bench}.json")))
    out = {}
    for k, v in (d.get("per_problem") or {}).items():
        if isinstance(v, list) and v:
            out[k] = sum(1.0 if x else 0.0 for x in v) / len(v)
        elif isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
    return out


def ci(A, B, seed, reps=10000):
    ids = sorted(set(A) & set(B))
    d = [B[i] - A[i] for i in ids]
    rnd = random.Random(seed)
    n = len(d)
    vals = sorted(100.0 * sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return vals[int(0.025 * reps)], vals[int(0.975 * reps)]


print("=== seed stability: IFBench control -> edit, accuracy CI lower bound ===")
A, B = scores("Rcc_0.1", "ifbench"), scores("Bcc_0.1", "ifbench")
los = []
for s in (1, 7, 42, 100, 2026, 20261107, 999983, 123456, 31337, 88888):
    lo, hi = ci(A, B, s)
    los.append(lo)
    print(f"  seed {s:>9}: [{lo:+.2f}, {hi:+.2f}]{'   <-- crosses 0' if lo <= 0 else ''}")
neg = sum(1 for x in los if x <= 0)
print(f"  lower bound range: {min(los):+.2f} .. {max(los):+.2f}")
print(f"  seeds where the CI includes 0: {neg} of {len(los)}")
print(f"  VERDICT: {'seed-sensitive -> must be reported as borderline, not a win' if neg else 'stable above 0'}")
print()

# Same check on the two clauses the verdict actually rests on, to show they are NOT borderline.
print("=== for contrast: the two clauses the PASS rests on ===")
for bench, label in (("gpqa_diamond", "GPQA"), ("lcb_v6", "LCB")):
    A2, B2 = scores("Rcc_0.1", bench), scores("Bcc_0.1", bench)
    lo2 = [ci(A2, B2, s)[0] for s in (1, 42, 2026, 20261107, 88888)]
    print(f"  {label:5s} ctrl->edit lower bound across 5 seeds: {min(lo2):+.2f} .. {max(lo2):+.2f}  "
          f"({'STABLE above 0' if min(lo2) > 0 else 'crosses 0'})")

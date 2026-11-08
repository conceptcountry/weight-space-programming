#!/usr/bin/env python3
"""Record the 128k robustness run's live coverage into results.json.

The robustness check re-runs, at a 128k cap, every sample that hit the 32k cap,
so that truncation's effect can be bounded rather than assumed. It is a
pre-registered robustness check, never part of the success criterion.

An earlier attempt died of host memory exhaustion after 3 samples. This records
whatever the current run has actually produced, so the paper quotes a measured
coverage instead of a remembered one. Rerun it whenever the run advances; the
paper's constants regenerate from whatever this writes.

Coverage is counted in real (id, sample) pairs. Placeholder rows carry no text
and are explicitly NOT counted as done: the run's own resume logic treats them
as complete, which is a defect being tracked separately, and a coverage figure
that silently included them would overstate what was measured.
"""
import argparse
import json
import pathlib
import subprocess
import sys

SSH = [
    "ssh", "-i", str(pathlib.Path.home() / ".ssh/s2_dgx_ed25519"),
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
    "somewheresystems@10.88.0.22",
]

REMOTE = r"""
import json, os
R = os.path.expanduser("~/wsp")
arm = "Bcc_0.1"
out = f"{R}/results/aC_{arm}_gpqa_g_{arm}.jsonl"
prompts = f"{R}/data/gpqa_g_{arm}.jsonl"

n_prompts = sum(1 for l in open(prompts) if l.strip())
rows = [json.loads(l) for l in open(out)] if os.path.exists(out) else []

real, placeholder = set(), set()
for r in rows:
    k = (r.get("id"), r.get("sample"))
    if r.get("placeholder") or not r.get("text"):
        placeholder.add(k)
    else:
        real.add(k)

print(json.dumps({
    "pairs_planned": n_prompts * 2,
    "ids_planned": n_prompts,
    "pairs_done": len(real),
    "pairs_placeholder": len(placeholder - real),
    "tokens_generated": sum(r.get("ntok", 0) for r in rows if r.get("text")),
    "cap": 131072,
}))
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    a = ap.parse_args()

    p = subprocess.run(SSH + ["python3", "-"],
                       input=REMOTE,
                       capture_output=True, text=True, timeout=90)
    if p.returncode != 0:
        print(f"  remote read failed: {p.stderr.strip()[:200]}", file=sys.stderr)
        return 1

    live = json.loads(p.stdout.strip().splitlines()[-1])
    done, planned = live["pairs_done"], live["pairs_planned"]
    live["complete"] = done >= planned
    live["status"] = "complete" if live["complete"] else "in progress"

    f = pathlib.Path(a.results)
    res = json.loads(f.read_text())
    res["robustness_128k"] = live
    f.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")

    print(f"  cap {live['cap']}: {done}/{planned} pairs over {live['ids_planned']} ids"
          f"  ({live['tokens_generated']:,} tokens, {live['status']})")
    if live["pairs_placeholder"]:
        print(f"  {live['pairs_placeholder']} placeholder rows not counted as done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

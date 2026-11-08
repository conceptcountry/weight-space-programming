#!/usr/bin/env python3
"""Emit results.json for the WSP paper from graded + raw evaluation artifacts.

Every statistic the paper quotes is computed here. Nothing is transcribed by hand.

Critical: GPQA raw files are contaminated by avg@4 follow-up runs that append
samples 2 and 3 into the same file (base 406 rows, Bcc 792, Rcc 396). Any read of
a raw GPQA file MUST filter to samples 0,1 or it mixes 2-sample and 4-sample
problems in the per-problem mean. This bit three separate consumers during the
confirmation; see RESULTS2.md. IFBench and LCB are sample-0 only, so the filter
is a no-op there but is applied uniformly anyway.
"""
import json, os, random, statistics, hashlib

WSP = os.path.expanduser("~/wsp")
R = os.path.join(WSP, "results")
S4 = os.path.join(R, "s4")

ARMS = ["base", "Bcc_0.1", "Rcc_0.1"]
GPQA_SAMPLES = {0, 1}
BOOT = 10000
SEED = 20261107


def find(*cands):
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def raw_path(arm, bench):
    return find(os.path.join(R, f"aC_{arm}_{bench}.jsonl"),
                os.path.join(S4, f"aC_{arm}_{bench}.jsonl"))


def graded_path(arm, bench):
    return find(os.path.join(R, f"g_{arm}_{bench}.json"),
                os.path.join(S4, f"g_{arm}_{bench}.json"))


def sha12(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def load_raw(arm, bench, filter_samples=True):
    """id -> {sample: row}, optionally filtered to samples 0,1."""
    p = raw_path(arm, bench)
    if not p:
        return None, None
    per = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            s = r.get("sample", 0)
            if filter_samples and bench.startswith("gpqa") and s not in GPQA_SAMPLES:
                continue
            per.setdefault(r["id"], {})[s] = r
    return per, p


def per_problem_scores(arm, bench):
    """id -> mean score in [0,1], from the graded json (authoritative for correctness)."""
    p = graded_path(arm, bench)
    if not p:
        return None, None
    d = json.load(open(p))
    pp = d.get("per_problem") or {}
    out = {}
    for k, v in pp.items():
        if isinstance(v, list):
            if not v:
                continue
            out[k] = sum(1.0 if x else 0.0 for x in v) / len(v)
        elif isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out, p


def boot_ci(diffs, stat, reps=BOOT, seed=SEED):
    rnd = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return None, None
    vals = []
    for _ in range(reps):
        samp = [diffs[rnd.randrange(n)] for _ in range(n)]
        vals.append(stat(samp))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps)]


def paired_acc(A, B):
    ids = sorted(set(A) & set(B))
    d = [B[i] - A[i] for i in ids]
    mean = 100.0 * sum(d) / len(d)
    lo, hi = boot_ci(d, lambda s: 100.0 * sum(s) / len(s))
    return {
        "n": len(ids),
        "a": round(100.0 * sum(A[i] for i in ids) / len(ids), 2),
        "b": round(100.0 * sum(B[i] for i in ids) / len(ids), 2),
        "delta": round(mean, 2),
        "ci": [round(lo, 2), round(hi, 2)],
        "better": sum(1 for x in d if x > 0),
        "worse": sum(1 for x in d if x < 0),
        "same": sum(1 for x in d if x == 0),
    }


def tokens_and_trunc(arm, bench):
    """Mean tokens per problem (averaging samples within a problem first) + truncation count."""
    per, p = load_raw(arm, bench)
    if not per:
        return None
    means, trunc, n_rows = {}, 0, 0
    for pid, samples in per.items():
        toks = []
        for r in samples.values():
            n_rows += 1
            nt = r.get("ntok")
            if isinstance(nt, (int, float)):
                toks.append(float(nt))
            if r.get("finish") == "length":
                trunc += 1
        if toks:
            means[pid] = sum(toks) / len(toks)
    return {"per_problem": means, "trunc": trunc, "rows": n_rows,
            "path": os.path.relpath(p, WSP), "sha12": sha12(p)}


def paired_tokens(A, B):
    """Paired bootstrap of the mean-token delta as a percentage.

    Matches scripts/paired.py in the evaluation harness, the method behind every
    published token figure: resample the per-problem DIFFERENCES and divide by a
    fixed baseline mean, rather than resampling both arms and re-deriving the
    ratio. Resampling both narrows the interval (it cancels shared problem-level
    variance), so using it here would silently publish tighter CIs than the
    confirmation reported. Keep these two implementations in agreement.
    """
    ids = sorted(set(A["per_problem"]) & set(B["per_problem"]))
    a = [A["per_problem"][i] for i in ids]
    b = [B["per_problem"][i] for i in ids]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    diffs = [y - x for x, y in zip(a, b)]
    pct = 100.0 * (sum(diffs) / len(diffs)) / ma
    rnd = random.Random(SEED)
    n = len(diffs)
    vals = sorted(100.0 * (sum(diffs[rnd.randrange(n)] for _ in range(n)) / n) / ma
                  for _ in range(BOOT))
    lo, hi = vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT)]
    return {"n": len(ids), "a": round(ma, 1), "b": round(mb, 1),
            "pct": round(pct, 1), "ci": [round(lo, 1), round(hi, 1)],
            "trunc_a": A["trunc"], "trunc_b": B["trunc"]}


out = {"generated_utc": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
       "bootstrap_reps": BOOT, "bootstrap_seed": SEED,
       "gpqa_samples_filter": sorted(GPQA_SAMPLES),
       "benchmarks": {}, "provenance": {}, "census": {}}

for bench, label in (("gpqa_diamond", "GPQA Diamond"), ("ifbench", "IFBench"), ("lcb_v6", "LiveCodeBench v6")):
    acc, gp = {}, {}
    for arm in ARMS:
        s, p = per_problem_scores(arm, bench)
        if s is None:
            print(f"  WARN no graded file: {arm} {bench}")
            continue
        acc[arm] = s
        gp[arm] = {"path": os.path.relpath(p, WSP), "sha12": sha12(p), "ids": len(s)}
    if len(acc) < 3:
        continue

    tok = {}
    for arm in ARMS:
        t = tokens_and_trunc(arm, bench)
        if t:
            tok[arm] = t

    entry = {"label": label, "n_ids": len(acc["base"]),
             "acc": {a: round(100.0 * sum(v.values()) / len(v), 2) for a, v in acc.items()},
             "paired": {
                 "base_to_edit": paired_acc(acc["base"], acc["Bcc_0.1"]),
                 "ctrl_to_edit": paired_acc(acc["Rcc_0.1"], acc["Bcc_0.1"]),
                 "base_to_ctrl": paired_acc(acc["base"], acc["Rcc_0.1"]),
             }}
    if len(tok) == 3:
        entry["tokens"] = {
            "mean": {a: round(sum(t["per_problem"].values()) / len(t["per_problem"]), 1) for a, t in tok.items()},
            "trunc": {a: t["trunc"] for a, t in tok.items()},
            "base_to_edit": paired_tokens(tok["base"], tok["Bcc_0.1"]),
            "ctrl_to_edit": paired_tokens(tok["Rcc_0.1"], tok["Bcc_0.1"]),
            "base_to_ctrl": paired_tokens(tok["base"], tok["Rcc_0.1"]),
        }
        entry["provenance_raw"] = {a: {"path": t["path"], "sha12": t["sha12"], "rows": t["rows"]} for a, t in tok.items()}
    entry["provenance_graded"] = gp
    out["benchmarks"][bench] = entry

# GPQA sample census: proves the avg@4 filter is doing its job.
for arm in ARMS:
    unf, p = load_raw(arm, "gpqa_diamond", filter_samples=False)
    fil, _ = load_raw(arm, "gpqa_diamond", filter_samples=True)
    if unf and fil:
        import collections
        out["census"][arm] = {
            "rows_unfiltered": sum(len(v) for v in unf.values()),
            "rows_filtered": sum(len(v) for v in fil.values()),
            "sample_counts_filtered": dict(collections.Counter(len(v) for v in fil.values())),
            "sample_counts_unfiltered": dict(collections.Counter(len(v) for v in unf.values())),
        }

print(json.dumps(out, indent=2, sort_keys=True))

#!/usr/bin/env python3.12
"""Fake-recall at real-recall floors, swept from a results_gsd.txt (test_video_image_batch.py output).

Decision rule: predict FAKE iff fake_score >= threshold. As the threshold rises, real-recall rises and
fake-recall falls (monotone). For each real-recall FLOOR we pick the LOWEST threshold whose real-recall
>= floor (which maximises fake-recall while meeting the floor) and report the fake-recall there.

  python3.12 fake_recall_at_floors.py runs/floor_eval/results_gsd.txt [--floors 0.80 0.85 0.90 0.95 0.98]
"""
import argparse
import re

AP = argparse.ArgumentParser()
AP.add_argument("results")
AP.add_argument("--floors", type=float, nargs="*", default=[0.80, 0.85, 0.90, 0.95, 0.98])
args = AP.parse_args()

pat = re.compile(r"truth=(\w+).*?fake=([0-9.]+)")
real, fake = [], []
for ln in open(args.results):
    m = pat.search(ln)
    if not m:
        continue
    t, s = m.group(1), float(m.group(2))
    (real if t == "real" else fake if t == "fake" else []).append(s) if t in ("real", "fake") else None

real.sort(); fake.sort()
nr, nf = len(real), len(fake)
import bisect


def real_recall(t):                       # fraction of reals scored below t (predicted real)
    return bisect.bisect_left(real, t) / nr


def fake_recall(t):                        # fraction of fakes scored >= t (predicted fake)
    return (nf - bisect.bisect_left(fake, t)) / nf


# candidate thresholds: every distinct real score (+ a hair above) gives every attainable real-recall
cands = sorted(set(real + fake + [0.0, 1.0 + 1e-9]))
print(f"n_real={nr}  n_fake={nf}\n")
print(f"{'real-floor':>10} | {'threshold':>9} | {'real-recall':>11} | {'fake-recall':>11}")
print("-" * 52)
for R in args.floors:
    # lowest threshold achieving real_recall >= R
    t_hit = next((t for t in cands if real_recall(t) >= R), 1.0 + 1e-9)
    print(f"{R*100:9.0f}% | {t_hit:9.4f} | {real_recall(t_hit)*100:10.2f}% | {fake_recall(t_hit)*100:10.2f}%")

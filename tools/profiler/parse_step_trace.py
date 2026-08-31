#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flatten vLLM XPU per-step TRACE logs into CSV/JSON.

See docs/contributing/profiling.md ("Per-Step Tracing on XPU") for how to
produce these logs. Handles the format emitted with TRACE=1:

    m = 3, step = 54:
        total_requests_num: 3
        total_num_scheduled_tokens: 3
        request id: 1-aaa2f954
            context_len: 64
            num_scheduled_token: 1
        ...
    step = 53, m = 4, forward device: 29.995 ms   <- belongs to step 53
    preprocess time: 8.487 ms
    forward time: 33.128 ms
    postprocess time: 1.376 ms
    execute_model time: 42.991 ms
    sample_tokens time: 0.562 ms
    step time (worker): 43.699 ms

Two things make this non-trivial:

* ``forward device`` is measured with XPU events and drained non-blocking, so
  it is printed one or more steps after the step it describes. It is matched
  back by the ``step =`` tag it carries, not by position.
* ``m = 0`` steps are scheduler no-ops that never run a forward, so they have
  a header but no timings. They are dropped by default.

Usage:
    cd tools/profiler
    python parse_step_trace.py server.log -o steps.csv
    python parse_step_trace.py server.log -o steps.csv --json steps.json
    python parse_step_trace.py server.log -o steps.csv --keep-empty --min-step 10
"""

import argparse
import csv
import json
import re
import sys

# Strip "(EngineCore pid=245) INFO 08-25 10:31:19 [xpu_model_runner.py:127] "
PREFIX = re.compile(r"^.*?\[[\w.]+:\d+\]\s*")

HEADER = re.compile(r"^m = (\d+), step = (-?\d+):")
DEVICE = re.compile(r"^step = (-?\d+), m = (\d+), forward device: ([\d.]+) ms")
PHASE = re.compile(
    r"^(preprocess|forward|postprocess|execute_model|sample_tokens) time: ([\d.]+) ms"
)
STEP_TOTAL = re.compile(r"^step time \(worker\): ([\d.]+) ms")
TOTAL_REQS = re.compile(r"^total_requests_num: (\d+)")
TOTAL_TOKS = re.compile(r"^total_num_scheduled_tokens: (\d+)")
CTX_LEN = re.compile(r"^context_len: (\d+)")
NUM_SCHED = re.compile(r"^num_scheduled_token: (\d+)")

PHASE_FIELD = {
    "preprocess": "preprocess_ms",
    "forward": "forward_ms",
    "postprocess": "postprocess_ms",
    "execute_model": "execute_model_ms",
    "sample_tokens": "sample_tokens_ms",
}

FIELDS = [
    "step",
    "m",
    "num_requests",
    "prefill_reqs",
    "decode_reqs",
    "prefill_tokens",
    "decode_tokens",
    "preprocess_ms",
    "forward_ms",
    "postprocess_ms",
    "execute_model_ms",
    "sample_tokens_ms",
    "step_worker_ms",
    "forward_device_ms",
    "device_over_host",
    "context_lens",
    "num_scheduled_tokens",
]


def new_step(m, step):
    return {
        "step": step,
        "m": m,
        "num_requests": None,
        "prefill_reqs": 0,
        "decode_reqs": 0,
        "prefill_tokens": 0,
        "decode_tokens": 0,
        "preprocess_ms": None,
        "forward_ms": None,
        "postprocess_ms": None,
        "execute_model_ms": None,
        "sample_tokens_ms": None,
        "step_worker_ms": None,
        "forward_device_ms": None,
        "device_over_host": None,
        "context_lens": [],
        "num_scheduled_tokens": [],
    }


def parse(lines):
    """Return (steps_in_order, warnings)."""
    steps = {}
    order = []
    cur = None
    # forward-device lines seen before their step header (shouldn't happen, but
    # be safe) are parked here.
    orphan_device = {}
    warnings = []

    for raw in lines:
        line = PREFIX.sub("", raw).strip()
        if not line:
            continue

        m = HEADER.match(line)
        if m:
            mm, step = int(m.group(1)), int(m.group(2))
            if step in steps:
                warnings.append(f"duplicate header for step {step}; keeping first")
                cur = steps[step]
                continue
            cur = new_step(mm, step)
            steps[step] = cur
            order.append(step)
            if step in orphan_device:
                cur["forward_device_ms"] = orphan_device.pop(step)
            continue

        # Device timing carries its own step tag and is printed out of order.
        d = DEVICE.match(line)
        if d:
            step, dev = int(d.group(1)), float(d.group(3))
            target = steps.get(step)
            if target is None:
                orphan_device[step] = dev
            else:
                target["forward_device_ms"] = dev
            continue

        if cur is None:
            continue

        p = PHASE.match(line)
        if p:
            cur[PHASE_FIELD[p.group(1)]] = float(p.group(2))
            continue

        t = STEP_TOTAL.match(line)
        if t:
            cur["step_worker_ms"] = float(t.group(1))
            continue

        r = TOTAL_REQS.match(line)
        if r:
            cur["num_requests"] = int(r.group(1))
            continue

        c = CTX_LEN.match(line)
        if c:
            cur["context_lens"].append(int(c.group(1)))
            continue

        n = NUM_SCHED.match(line)
        if n:
            tok = int(n.group(1))
            cur["num_scheduled_tokens"].append(tok)
            if tok > 1:
                cur["prefill_reqs"] += 1
                cur["prefill_tokens"] += tok
            else:
                cur["decode_reqs"] += 1
                cur["decode_tokens"] += tok
            continue

    if orphan_device:
        warnings.append(
            f"{len(orphan_device)} device timing(s) with no matching step header: "
            f"{sorted(orphan_device)[:5]}"
        )

    return [steps[s] for s in order], warnings


def finalize(rows, keep_empty, min_step):
    """Validate, derive ratios, and drop steps the caller does not want."""
    out, warnings = [], []
    missing_device = 0

    for r in rows:
        if r["m"] == 0 and not keep_empty:
            continue
        if min_step is not None and r["step"] < min_step:
            continue

        toks = r["num_scheduled_tokens"]
        if toks and sum(toks) != r["m"]:
            warnings.append(
                f"step {r['step']}: scheduled tokens sum {sum(toks)} != m {r['m']}"
            )
        if r["num_requests"] is not None and len(toks) != r["num_requests"]:
            warnings.append(
                f"step {r['step']}: {len(toks)} request entries != "
                f"total_requests_num {r['num_requests']}"
            )

        dev, host = r["forward_device_ms"], r["forward_ms"]
        if dev is None:
            missing_device += 1
        elif host:
            r["device_over_host"] = round(dev / host, 4)

        r["context_lens"] = ";".join(str(x) for x in r["context_lens"])
        r["num_scheduled_tokens"] = ";".join(str(x) for x in toks)
        out.append(r)

    if missing_device:
        warnings.append(
            f"{missing_device} step(s) without device timing "
            "(normal for the last few steps: events are drained lazily)"
        )
    return out, warnings


def summarize(rows):
    """Median per-phase timings, split by whether the step contained prefill."""

    def med(vals):
        vals = sorted(v for v in vals if v is not None)
        if not vals:
            return None
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

    groups = {
        "decode-only": [r for r in rows if r["prefill_reqs"] == 0],
        "with-prefill": [r for r in rows if r["prefill_reqs"] > 0],
    }
    lines = []
    hdr = (
        f"{'group':<14}{'n':>5}{'m_med':>9}{'pre':>9}{'fwd':>9}"
        f"{'post':>9}{'sample':>9}{'step':>9}{'fwd_dev':>9}{'dev/host':>10}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for name, g in groups.items():
        if not g:
            continue
        lines.append(
            f"{name:<14}{len(g):>5}"
            f"{med([r['m'] for r in g]) or 0:>9.0f}"
            f"{med([r['preprocess_ms'] for r in g]) or 0:>9.3f}"
            f"{med([r['forward_ms'] for r in g]) or 0:>9.3f}"
            f"{med([r['postprocess_ms'] for r in g]) or 0:>9.3f}"
            f"{med([r['sample_tokens_ms'] for r in g]) or 0:>9.3f}"
            f"{med([r['step_worker_ms'] for r in g]) or 0:>9.3f}"
            f"{med([r['forward_device_ms'] for r in g]) or 0:>9.3f}"
            f"{med([r['device_over_host'] for r in g]) or 0:>10.3f}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Flatten vLLM XPU per-step TRACE logs into CSV/JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1],
    )
    ap.add_argument("log_file", help="vLLM server/offline log with TRACE=1 output")
    ap.add_argument("-o", "--output", default="steps.csv", help="CSV output path")
    ap.add_argument("--json", dest="json_output", help="also write JSON here")
    ap.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep m=0 scheduler no-op steps (dropped by default)",
    )
    ap.add_argument(
        "--min-step",
        type=int,
        help=(
            "drop steps below this index (e.g. to skip warmup). Note that "
            "absolute step numbers are not stable across runs because idle "
            "m=0 steps between requests also advance the counter; prefer "
            "skipping warmup at the source via NUM_WARMUP_STEPS."
        ),
    )
    ap.add_argument("--quiet", action="store_true", help="suppress the summary")
    args = ap.parse_args()

    with open(args.log_file, errors="ignore") as f:
        rows, warnings = parse(f)

    if not rows:
        sys.exit(
            "No traced steps found. Was the server run with TRACE=1 and "
            "VLLM_USE_V2_MODEL_RUNNER=0? The V2 runner silently ignores these hooks."
        )

    rows, more = finalize(rows, args.keep_empty, args.min_step)
    warnings += more

    if not rows:
        sys.exit("All steps were filtered out; relax --min-step/--keep-empty.")

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(rows, f, indent=2)

    for msg in warnings:
        print(f"warning: {msg}", file=sys.stderr)

    print(f"Wrote {len(rows)} steps to {args.output}")
    if args.json_output:
        print(f"Wrote JSON to {args.json_output}")
    if not args.quiet:
        print()
        print(summarize(rows))
        print()
        print(
            "note: fwd_dev is the forward's device *span* (XPU events), not kernel\n"
            "      busy time. On a host-bound run the two can differ several-fold;\n"
            "      use PROFILE=1 to get busy time."
        )


if __name__ == "__main__":
    main()

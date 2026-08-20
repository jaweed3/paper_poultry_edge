#!/usr/bin/env python3
"""
Profiling Script for Level 2 Bottleneck Evidence
=================================================
1. Counts QuantizeLinear / DequantizeLinear nodes per model
2. Runs ORT profiling for INT8 models (MobileNetV2, ShuffleNetV2, EfficientNet-B0)
3. Saves JSON artifact for paper supplementary

Target: RPi5, batch=1, threads=1, 10 runs + 5 warmup
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import onnxruntime as ort

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("[WARN] onnx package not installed. Install: pip install onnx")


# ─── Config ──────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results/profiling")
PROFILING_RUNS = 10
WARMUP_RUNS = 5
BATCH_SIZE = 1
NUM_THREADS = 1

INT8_MODELS = [
    ("mobilenetv2", "models/ptq/mobilenetv2_int8.onnx"),
    ("shufflenetv2", "models/ptq/shufflenetv2_int8.onnx"),
    ("efficientnet_b0", "models/ptq/efficientnet_b0_int8.onnx"),
]

FP32_MODELS = [
    ("mobilenetv2", "models/onnx/mobilenetv2_fp32.onnx"),
    ("shufflenetv2", "models/onnx/shufflenetv2_fp32.onnx"),
    ("efficientnet_b0", "models/onnx/efficientnet_b0_fp32.onnx"),
]


# ─── Node Count Analysis ─────────────────────────────────────────────────────

def count_quant_nodes(onnx_path):
    """
    Count QuantizeLinear, DequantizeLinear, and total op nodes in an ONNX model.
    """
    result = {
        "model_path": str(onnx_path),
        "model_size_mb": round(os.path.getsize(onnx_path) / (1024 * 1024), 3),
        "total_nodes": 0,
        "quantize_linear": 0,
        "dequantize_linear": 0,
        "op_counts": {},
    }

    if not HAS_ONNX:
        # Fallback: use onnxruntime to inspect
        return _count_nodes_ort(onnx_path, result)

    model = onnx.load(str(onnx_path))
    result["total_nodes"] = len(model.graph.node)

    for node in model.graph.node:
        op = node.op_type
        result["op_counts"][op] = result["op_counts"].get(op, 0) + 1

        if op == "QuantizeLinear":
            result["quantize_linear"] += 1
        elif op == "DequantizeLinear":
            result["dequantize_linear"] += 1

    result["q_dq_total"] = result["quantize_linear"] + result["dequantize_linear"]
    result["q_dq_ratio"] = round(
        result["q_dq_total"] / max(result["total_nodes"], 1) * 100, 1
    )

    return result


def _count_nodes_ort(onnx_path, result):
    """Fallback node counting using ORT graph inspection."""
    try:
        import onnxruntime as ort
        # ORT doesn't directly expose node counts, but we can read the serialized model
        # Use onnxruntime to get basic info
        session = ort.InferenceSession(str(onnx_path))
        result["total_nodes"] = "N/A (onnx package required)"
        result["note"] = "Install 'onnx' package for detailed node analysis"
        del session
    except Exception as e:
        result["error"] = str(e)
    return result


def print_node_table(all_node_counts):
    """Print node count comparison table."""
    print("\n" + "=" * 90)
    print("  Q/DQ NODE COUNT ANALYSIS")
    print("=" * 90)
    print(f"{'Model':<18} {'Quant':<6} {'Total Nodes':<14} {'QL':<8} {'DQL':<8} {'Q+DQL':<8} {'Q+DQL %':<10} {'Size (MB)':<10}")
    print("-" * 90)

    for r in all_node_counts:
        name = Path(r["model_path"]).stem.replace("_int8", "").replace("_fp32", "")
        quant = "INT8" if "ptq" in r["model_path"] or "int8" in r["model_path"] else "FP32"
        ql = r.get("quantize_linear", 0)
        dql = r.get("dequantize_linear", 0)
        total = r.get("total_nodes", "?")
        qdq = r.get("q_dq_total", 0)
        ratio = r.get("q_dq_ratio", 0)

        print(
            f"{name:<18} {quant:<6} {str(total):<14} {ql:<8} {dql:<8} {qdq:<8} {ratio:<10}% {r['model_size_mb']:<10}"
        )

    print("=" * 90)


# ─── ORT Profiling ───────────────────────────────────────────────────────────

def run_ort_profile(model_name, onnx_path, num_threads, batch_size, num_runs, warmup):
    """
    Run ORT inference with profiling enabled.
    Returns per-node timing breakdown.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.enable_profiling = True
    opts.profile_file_prefix = str(RESULTS_DIR / f"profile_{model_name}")
    opts.log_severity_level = 3

    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = tuple(
        batch_size if (isinstance(d, str) or d is None or d <= 0) else d
        for d in input_meta.shape
    )
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    # Profiled runs
    latencies = []
    for i in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    # End profiling - this generates the profile JSON
    profile_file = session.end_profiling()

    # Parse the profile JSON to extract node timing
    profile_data = _parse_profile(profile_file)

    return {
        "model": model_name,
        "profile_file": profile_file,
        "latency_mean_ms": round(float(np.mean(latencies)), 2),
        "latency_std_ms": round(float(np.std(latencies)), 2),
        "runs": num_runs,
        "threads": num_threads,
        "batch_size": batch_size,
        "node_timing": profile_data,
    }


def _parse_profile(profile_file):
    """Parse ORT profiling JSON to extract node timing breakdown."""
    try:
        with open(profile_file) as f:
            data = json.load(f)
    except Exception as e:
        return {"error": str(e), "file": profile_file}

    # ORT profiling format: list of events
    # Each event has: cat, pid, tid, ts, dur, name, args
    if not isinstance(data, list):
        # Try to extract from the dict format
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        else:
            return {"error": "Unexpected profile format", "file": profile_file}

    # Aggregate timing by operator type
    op_timing = {}
    total_dur = 0

    for event in data:
        if event.get("cat") == "Node" and "dur" in event:
            name = event.get("name", "unknown")
            dur = event.get("dur", 0)  # microseconds

            # Extract op type from node name
            # Typical format: "op_name (OpType)" or just "OpType"
            op_type = name
            if "(" in name and ")" in name:
                op_type = name.split("(")[-1].rstrip(")")
            elif "/" in name:
                op_type = name.split("/")[-1]

            if op_type not in op_timing:
                op_timing[op_type] = {"total_us": 0, "count": 0}
            op_timing[op_type]["total_us"] += dur
            op_timing[op_type]["count"] += 1
            total_dur += dur

    # Sort by total duration
    sorted_ops = sorted(op_timing.items(), key=lambda x: x[1]["total_us"], reverse=True)

    # Calculate percentages and convert to ms
    breakdown = []
    for op_type, timing in sorted_ops:
        pct = round(timing["total_us"] / max(total_dur, 1) * 100, 1)
        breakdown.append({
            "op_type": op_type,
            "total_us": timing["total_us"],
            "total_ms": round(timing["total_us"] / 1000, 2),
            "count": timing["count"],
            "percentage": pct,
        })

    return {
        "total_duration_us": total_dur,
        "total_duration_ms": round(total_dur / 1000, 2),
        "operators": breakdown,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  LEVEL 2 PROFILING — Q/DQ Node Count + ORT Profiling")
    print(f"  RPi5 | {NUM_THREADS} thread | batch {BATCH_SIZE} | {PROFILING_RUNS} runs")
    print("=" * 70)

    # ── Part 1: Node Count Analysis ──────────────────────────────────────
    print("\n[1/2] Counting Q/DQ nodes per model...")
    all_node_counts = []

    print(f"\n{'Model':<18} {'QL':<8} {'DQL':<8} {'Q+DQL':<8} {'Total':<10} {'Ratio':<8}")
    print("-" * 60)

    for model_name, onnx_path in INT8_MODELS + FP32_MODELS:
        if not os.path.exists(onnx_path):
            print(f"[SKIP] {onnx_path} not found")
            continue
        r = count_quant_nodes(onnx_path)
        all_node_counts.append(r)

        name = model_name
        quant = "INT8" if "int8" in onnx_path else "FP32"
        ql = r.get("quantize_linear", 0)
        dql = r.get("dequantize_linear", 0)
        qdq = r.get("q_dq_total", 0)
        total = r.get("total_nodes", 0)
        ratio = r.get("q_dq_ratio", 0)

        print(f"{name} ({quant})") if False else print(
            f"{name + ' ' + quant:<18} {ql:<8} {dql:<8} {qdq:<8} {str(total):<10} {ratio}%"
        )

    print_node_table(all_node_counts)

    # ── Part 2: ORT Profiling (INT8 models only) ────────────────────────
    print("\n[2/2] Running ORT profiling on INT8 models...")
    all_profiles = []

    for model_name, onnx_path in INT8_MODELS:
        if not os.path.exists(onnx_path):
            print(f"[SKIP] {onnx_path} not found")
            continue

        print(f"\n  Profiling {model_name} INT8 ({PROFILING_RUNS} runs)...")
        profile = run_ort_profile(
            model_name=model_name,
            onnx_path=onnx_path,
            num_threads=NUM_THREADS,
            batch_size=BATCH_SIZE,
            num_runs=PROFILING_RUNS,
            warmup=WARMUP_RUNS,
        )
        all_profiles.append(profile)

        # Print top operators by time
        ops = profile["node_timing"].get("operators", [])
        print(f"    Latency: {profile['latency_mean_ms']} ms (std: {profile['latency_std_ms']} ms)")
        print(f"    Profile saved: {profile['profile_file']}")
        print(f"    Top 5 operators by time:")
        for i, op in enumerate(ops[:5]):
            print(f"      {i+1}. {op['op_type']:<30} {op['total_ms']:>8.2f} ms  ({op['percentage']}%)")

    # ── Save Artifact ────────────────────────────────────────────────────
    artifact = {
        "experiment": "Level 2 Profiling — Q/DQ Node Count + ORT Operator Timing",
        "timestamp": datetime.now().isoformat(),
        "platform": "Raspberry Pi 5 (BCM2712 Cortex-A76)",
        "config": {
            "num_threads": NUM_THREADS,
            "batch_size": BATCH_SIZE,
            "profiling_runs": PROFILING_RUNS,
            "warmup_runs": WARMUP_RUNS,
        },
        "node_counts": all_node_counts,
        "ort_profiles": [],
    }

    # Add profiles without raw file references
    for p in all_profiles:
        profile_summary = {
            "model": p["model"],
            "latency_mean_ms": p["latency_mean_ms"],
            "latency_std_ms": p["latency_std_ms"],
            "runs": p["runs"],
            "node_timing": p["node_timing"],
        }
        artifact["ort_profiles"].append(profile_summary)

    artifact_path = RESULTS_DIR / "level2_profiling_artifact.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\n[OK] Artifact saved: {artifact_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  KEY FINDINGS FOR PAPER")
    print("=" * 70)

    # Find INT8 models for comparison
    int8_counts = {r["model_path"]: r for r in all_node_counts if "int8" in r["model_path"]}

    for model_name, onnx_path in INT8_MODELS:
        if onnx_path not in int8_counts:
            continue
        r = int8_counts[onnx_path]
        profile = next((p for p in all_profiles if p["model"] == model_name), None)

        print(f"\n  {model_name} INT8:")
        print(f"    Q+DQ nodes: {r.get('q_dq_total', 0)} ({r.get('q_dq_ratio', 0)}% of total {r.get('total_nodes', 0)})")

        if profile and "operators" in profile.get("node_timing", {}):
            ops = profile["node_timing"]["operators"]
            dequant = next((o for o in ops if "DequantizeLinear" in o["op_type"]), None)
            if dequant:
                print(f"    DequantizeLinear time: {dequant['total_ms']} ms ({dequant['percentage']}% of total)")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

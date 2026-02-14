"""Parameter search for chunk_size and chunk_overlap."""
import subprocess
import json
import re
import sys
import os
from pathlib import Path
from itertools import product

CHUNK_SIZES = [256, 512, 768, 1024, 1536]
CHUNK_OVERLAPS = [0, 50, 100, 200]

PROJECT_DIR = Path(__file__).parent.parent
RESULTS = {}

def run_combo(chunk_size, chunk_overlap):
    storage_dir = f"./storage_{chunk_size}_{chunk_overlap}"
    env = os.environ.copy()
    env["CHUNK_SIZE"] = str(chunk_size)
    env["CHUNK_OVERLAP"] = str(chunk_overlap)
    env["CHROMA_PERSIST_DIR"] = storage_dir
    
    print(f"\n{'='*60}")
    print(f"Testing chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"{'='*60}")
    
    # Remove partial results to avoid resume interference
    partial = PROJECT_DIR / "eval" / "results_step3_bitext.partial.json"
    if partial.exists():
        partial.unlink()
    full = PROJECT_DIR / "eval" / "results_step3_bitext.json"
    if full.exists():
        full.unlink()
    
    # Step 1: Ingest
    print(f"  Ingesting into {storage_dir}...")
    r = subprocess.run(
        ["uv", "run", "python", "-m", "kbee.ingest", "--clear"],
        cwd=str(PROJECT_DIR), env=env, capture_output=True, text=True, timeout=120
    )
    if r.returncode != 0:
        print(f"  INGEST FAILED: {r.stderr[-500:]}")
        return None
    
    # Extract chunk count from ingest output
    chunks = None
    for line in r.stderr.split("\n"):
        m = re.search(r"Ingested (\d+) chunks", line)
        if m:
            chunks = int(m.group(1))
    print(f"  Chunks: {chunks}")
    
    # Step 2: Eval
    print(f"  Running eval...")
    r = subprocess.run(
        ["uv", "run", "python", "eval/step3_bitext_eval.py"],
        cwd=str(PROJECT_DIR), env=env, capture_output=True, text=True, timeout=600
    )
    if r.returncode != 0:
        print(f"  EVAL FAILED: {r.stderr[-500:]}")
        return None
    
    # Parse retrieval accuracy from output
    accuracy = None
    for line in r.stdout.split("\n"):
        m = re.search(r"Retrieval accuracy:\s*(\d+)/(\d+)\s*\(([0-9.]+%)\)", line)
        if m:
            accuracy = f"{m.group(1)}/{m.group(2)} ({m.group(3)})"
            correct = int(m.group(1))
            total = int(m.group(2))
    
    # Also read from JSON
    if full.exists():
        data = json.loads(full.read_text())
        accuracy = data.get("retrieval_accuracy", accuracy)
    
    print(f"  Result: {accuracy}, chunks={chunks}")
    return {"accuracy": accuracy, "correct": correct, "total": total, "chunks": chunks}


def main():
    results_file = PROJECT_DIR / "eval" / "param_search_results.json"
    
    # Resume support
    if results_file.exists():
        all_results = json.loads(results_file.read_text())
    else:
        all_results = {}
    
    for cs, co in product(CHUNK_SIZES, CHUNK_OVERLAPS):
        if co >= cs:
            print(f"Skipping {cs}/{co} (overlap >= size)")
            continue
        key = f"{cs}_{co}"
        if key in all_results:
            print(f"Skipping {cs}/{co} (already done: {all_results[key]['accuracy']})")
            continue
        
        result = run_combo(cs, co)
        if result:
            all_results[key] = result
            results_file.write_text(json.dumps(all_results, indent=2))
    
    # Print summary table
    print("\n\n" + "="*70)
    print("PARAMETER SEARCH RESULTS")
    print("="*70)
    print(f"{'chunk_size':>12} {'overlap':>8} {'accuracy':>15} {'chunks':>8}")
    print("-"*50)
    for cs in CHUNK_SIZES:
        for co in CHUNK_OVERLAPS:
            key = f"{cs}_{co}"
            if key in all_results:
                r = all_results[key]
                print(f"{cs:>12} {co:>8} {r['accuracy']:>15} {r.get('chunks','?'):>8}")
    
    # Find best
    best_key = max(all_results, key=lambda k: (all_results[k]['correct'], -all_results[k].get('chunks', 999)))
    best = all_results[best_key]
    cs, co = best_key.split("_")
    print(f"\nBest: chunk_size={cs}, chunk_overlap={co} → {best['accuracy']}, {best.get('chunks')} chunks")


if __name__ == "__main__":
    main()

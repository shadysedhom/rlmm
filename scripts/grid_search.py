#!/usr/bin/env python3
"""Light-weight grid / random search driver for the simulator.

Usage (random):
  py scripts/grid_search.py \
        --template configs/default.yaml \
        --param_spec configs/param_search.yaml \
        --mode random --n_samples 400 --snapshots 1000 --workers 8

The param_spec YAML contains keys matching either top-level template keys or
nested strategy_kwargs entries, e.g.

k_vol: [0.5, 0.8, 1.0, 1.3, 1.6, 2.0]
gamma: [0.05, 0.1, 0.2]
size_base: [0.01, 0.02, 0.04]
order_ttl: [0.2, 0.3, 0.4]

For grid mode all Cartesian combinations are run; for random mode each sample
picks a random element from every list.
"""
from __future__ import annotations

import argparse, os, sys, tempfile, subprocess, uuid, json, csv, random, itertools, datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List

try:
    import yaml  # type: ignore
except ImportError:
    print("PyYAML is required. pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
RUN_SIM = ROOT / "run_simulator.py"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def save_yaml(d: Dict[str, Any], path: Path):
    with path.open("w") as f:
        yaml.dump(d, f)


def generate_combinations(param_spec: Dict[str, List[Any]], mode: str, n: int = 0):
    """Yield dicts of parameter overrides."""
    keys = list(param_spec.keys())
    lists = [param_spec[k] for k in keys]

    if mode == "grid":
        for combo in itertools.product(*lists):
            yield dict(zip(keys, combo))
    else:  # random
        for _ in range(n):
            yield {k: random.choice(v) for k, v in param_spec.items()}


def run_one(cfg_template: Dict[str, Any], overrides: Dict[str, Any], snapshots: int, base_out: Path) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex[:8]
    cfg = json.loads(json.dumps(cfg_template))  # deep copy via json

    # Split overrides into top-level and strategy_kwargs entries
    for k, v in overrides.items():
        if k in cfg:
            cfg[k] = v
        else:
            cfg.setdefault("strategy_kwargs", {})[k] = v

    # unique output dir / metrics path
    out_dir = base_out / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(out_dir)
    cfg["max_snapshots"] = snapshots

    tmp_cfg_path = base_out / f"tmp_{run_id}.yaml"
    save_yaml(cfg, tmp_cfg_path)

    cmd = [sys.executable, str(RUN_SIM), "--config", str(tmp_cfg_path), "--quiet"]
    subprocess.run(cmd, check=True)

    metrics_file = out_dir / f"{cfg['strategy']}_metrics.json"
    if not metrics_file.exists():
        return {"run_id": run_id, "error": "metrics_not_found"}
    with metrics_file.open() as f:
        metrics = json.load(f)
    metrics.update(overrides)
    metrics["run_id"] = run_id

    # Promote useful nested metrics for CSV convenience
    pa = metrics.get("pnl_attribution", {})
    am = metrics.get("additional_metrics", {})
    metrics["fees"] = pa.get("fees")
    metrics["adverse_selection"] = pa.get("adverse_selection")
    metrics["spread_capture"] = pa.get("spread_capture")
    metrics["forced_liquidations"] = am.get("forced_liquidations")

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True)
    p.add_argument("--param_spec", required=True)
    p.add_argument("--mode", choices=["grid", "random"], default="random")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--snapshots", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out_csv", default=f"grid_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    args = p.parse_args()

    template = load_yaml(Path(args.template))
    spec = load_yaml(Path(args.param_spec))

    base_out = Path(template.get("output_dir", "./results")) / "grid_search"
    base_out.mkdir(parents=True, exist_ok=True)

    combos = list(generate_combinations(spec, args.mode, args.n_samples))
    print(f"Generated {len(combos)} parameter sets")

    fieldnames = ["run_id", "final_pnl", "fees", "adverse_selection", "spread_capture", "forced_liquidations"] + list(spec.keys())
    rows: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_one, template, c, args.snapshots, base_out) for c in combos]
        for fut in as_completed(futures):
            res = fut.result()
            rows.append(res)
            print(f"finished {res.get('run_id')}, pnl={res.get('final_pnl')}")

    # write CSV
    with open(args.out_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {args.out_csv}")


if __name__ == "__main__":
    main() 
"""Multi-task × multi-seed benchmark runner.

Invokes `run_episode` from run_closed_loop.py for each (task, seed) pair,
aggregates success rates and failure-signal distributions, and emits both
a machine-readable JSON and a human-readable Markdown summary.

Usage:
    python -m closed_loop_cap.run_benchmark \
        --tasks beat_block_hammer shake_bottle \
        --seeds 0 1 2 3 4 \
        --config closed_loop_cap/configs/default.yaml

    # Resume — skip (task, seed) pairs whose report.json already exists
    python -m closed_loop_cap.run_benchmark --tasks ... --seeds ... --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.run_closed_loop import (  # noqa: E402
    EpisodeReport,
    _build_client,
    _load_config,
    run_episode,
)

logger = logging.getLogger("closed_loop_cap.benchmark")


# -----------------------------
# Aggregation
# -----------------------------


def _report_to_dict(report: EpisodeReport) -> dict[str, Any]:
    """Serialize EpisodeReport → dict with subtasks expanded."""
    base = {k: v for k, v in asdict(report).items() if k != "subtasks"}
    base["subtasks"] = [asdict(r) for r in report.subtasks]
    return base


def build_summary(reports_by_task: dict[str, list[dict]]) -> dict:
    """Collapse per-episode reports into per-task rows + grand totals."""
    rows = []
    grand = {
        "total": 0,
        "unstable_skipped": 0,
        "valid": 0,
        "success": 0,
        "signal_counter": Counter(),
    }

    for task, reps in reports_by_task.items():
        total = len(reps)
        unstable = sum(1 for r in reps if r.get("abort_reason") == "unstable_seed")
        valid = total - unstable
        success = sum(1 for r in reps if r.get("success"))

        signals: Counter[str] = Counter()
        abort_counter: Counter[str] = Counter()
        for r in reps:
            ar = r.get("abort_reason")
            if ar:
                # Normalize: strip trailing seed/id so "L1-S7_cascade: ..." groups together.
                abort_counter[ar.split(":", 1)[0].strip()] += 1
            for st in r.get("subtasks", []):
                for a in st.get("attempts", []):
                    if not a.get("ok") and a.get("signal_id"):
                        signals[a["signal_id"]] += 1

        durations = [float(r.get("duration_s") or 0.0) for r in reps]
        subtask_counts = [int(r.get("num_subtasks") or 0) for r in reps]

        rows.append(
            {
                "task": task,
                "total": total,
                "unstable_skipped": unstable,
                "valid": valid,
                "success": success,
                "success_rate": (success / valid) if valid else 0.0,
                "avg_duration_s": (sum(durations) / total) if total else 0.0,
                "avg_subtasks": (sum(subtask_counts) / total) if total else 0.0,
                "top_abort_reasons": abort_counter.most_common(5),
                "top_failure_signals": signals.most_common(8),
            }
        )

        grand["total"] += total
        grand["unstable_skipped"] += unstable
        grand["valid"] += valid
        grand["success"] += success
        grand["signal_counter"] += signals

    grand_rate = (grand["success"] / grand["valid"]) if grand["valid"] else 0.0
    return {
        "rows": rows,
        "grand_total": {
            "total": grand["total"],
            "unstable_skipped": grand["unstable_skipped"],
            "valid": grand["valid"],
            "success": grand["success"],
            "success_rate": grand_rate,
            "top_failure_signals": grand["signal_counter"].most_common(10),
        },
    }


def format_markdown(summary: dict) -> str:
    lines = ["# Benchmark Summary", ""]
    lines.append("## Per task")
    lines.append("")
    lines.append(
        "| Task | Total | Unstable | Valid | Success | Rate | Avg sub | Avg sec | Top signals |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in summary["rows"]:
        top_sig = ", ".join(f"{sid}({n})" for sid, n in r["top_failure_signals"][:3]) or "-"
        lines.append(
            f"| {r['task']} | {r['total']} | {r['unstable_skipped']} | {r['valid']} | "
            f"{r['success']} | {r['success_rate']:.0%} | "
            f"{r['avg_subtasks']:.1f} | {r['avg_duration_s']:.1f} | {top_sig} |"
        )

    g = summary["grand_total"]
    lines.append("")
    lines.append("## Grand total")
    lines.append("")
    lines.append(f"- Episodes requested: **{g['total']}**")
    lines.append(f"- Unstable skipped: **{g['unstable_skipped']}**")
    lines.append(f"- Valid (ran): **{g['valid']}**")
    lines.append(
        f"- Success: **{g['success']}/{g['valid']}** "
        f"= **{g['success_rate']:.1%}**"
    )
    if g["top_failure_signals"]:
        lines.append("")
        lines.append("### Top failure signals (all tasks)")
        for sid, n in g["top_failure_signals"]:
            lines.append(f"- `{sid}` — {n}")
    return "\n".join(lines) + "\n"


# -----------------------------
# Runner
# -----------------------------


def _load_existing_report(ep_dir: Path) -> dict | None:
    rp = ep_dir / "report.json"
    if not rp.is_file():
        return None
    try:
        with rp.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def run_benchmark(
    tasks: list[str],
    seeds: list[int] | dict[str, list[int]],
    config: dict,
    *,
    resume: bool,
    bench_dir: Path,
    planner_client: Any,
    codegen_client: Any,
    trials: int = 1,
    start_trial: int = 0,
    session: str = "",
) -> dict:
    """Run multi-task/multi-seed benchmark.

    seeds can be a single list (shared across tasks) or a per-task dict —
    the GT-seed path uses per-task dicts since feasible seeds differ by task.

    trials>1 (or start_trial>0) repeats each (task, seed) multiple times and
    writes per-trial artifacts under seed_N/trial_KKK/. trials==1 with
    start_trial==0 keeps the legacy seed_N/ layout.
    """
    bench_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(REPO_ROOT) / config.get("output_dir", "closed_loop_cap/output")
    from closed_loop_cap.paths import trial_dir as _td, default_session_id
    if not session:
        session = default_session_id()

    per_task_seeds: dict[str, list[int]] = (
        {t: list(seeds) for t in tasks} if isinstance(seeds, list)
        else {t: list(seeds.get(t, [])) for t in tasks}
    )

    start = start_trial if start_trial > 0 else 1
    trial_indices: list[int] = list(range(start, start + trials))

    reports_by_task: dict[str, list[dict]] = {t: [] for t in tasks}
    total = sum(len(v) for v in per_task_seeds.values()) * len(trial_indices)
    idx = 0
    t_bench = time.monotonic()

    for task in tasks:
        for seed in per_task_seeds[task]:
            for trial in trial_indices:
                idx += 1
                ep_dir = _td(output_root, session, task, seed, trial)
                tag = f"{task} seed={seed} trial={trial}"

                if resume:
                    existing = _load_existing_report(ep_dir)
                    if existing is not None:
                        logger.info("[%d/%d] %s — RESUME (cached)", idx, total, tag)
                        reports_by_task[task].append(existing)
                        continue

                logger.info("[%d/%d] %s — START", idx, total, tag)
                t0 = time.monotonic()
                try:
                    report = run_episode(
                        task, seed, config,
                        planner_client=planner_client,
                        codegen_client=codegen_client,
                        trial=trial, session=session,
                    )
                    reports_by_task[task].append(_report_to_dict(report))
                    dt = time.monotonic() - t0
                    logger.info(
                        "[%d/%d] %s — done success=%s in %.1fs",
                        idx, total, tag, report.success, dt,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("[%d/%d] %s — CRASH", idx, total, tag)
                    reports_by_task[task].append(
                        {
                            "task": task,
                            "seed": seed,
                            "trial": trial,
                            "success": False,
                            "abort_reason": f"exception: {type(exc).__name__}: {exc}",
                            "num_subtasks": 0,
                            "duration_s": time.monotonic() - t0,
                            "subtasks": [],
                        }
                    )

    summary = build_summary(reports_by_task)
    summary["elapsed_s"] = round(time.monotonic() - t_bench, 2)
    summary["tasks"] = tasks
    summary["seeds"] = (
        seeds if isinstance(seeds, list) else {t: per_task_seeds[t] for t in tasks}
    )
    summary["per_task_seeds"] = per_task_seeds
    summary["session"] = session

    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    md = format_markdown(summary)
    (bench_dir / "summary.md").write_text(md)
    (bench_dir / "per_episode.json").write_text(
        json.dumps(reports_by_task, indent=2, default=str)
    )
    return summary


# -----------------------------
# CLI
# -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop CaP multi-task benchmark")
    parser.add_argument("--tasks", nargs="+", required=True, help="Task names to evaluate")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit shared seed list (ignored when --seeds-from-gt is set).",
    )
    parser.add_argument(
        "--seeds-from-gt",
        action="store_true",
        help="Use the cached list of seeds whose hand-written play_once() succeeds, "
             "per task, from closed_loop_cap/output/_gt_seeds/<task>.json. "
             "Implies --auto-collect-seeds unless --no-auto-collect is passed.",
    )
    parser.add_argument(
        "--auto-collect-seeds", action="store_true",
        help="If a task has no cached GT seeds yet, run the collector before the benchmark.",
    )
    parser.add_argument(
        "--no-auto-collect", action="store_true",
        help="Override: do NOT run the collector; fail if any task has no cache.",
    )
    parser.add_argument(
        "--rebuild-gt-seeds", action="store_true",
        help="Ignore cached GT seeds and re-collect.",
    )
    parser.add_argument(
        "--gt-target-count", type=int, default=10,
        help="Target number of successful seeds to collect per task (default 10).",
    )
    parser.add_argument(
        "--gt-max-attempts", type=int, default=50,
        help="Max seeds to try while collecting (default 50).",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=None,
        help="After loading seeds for each task, use only the first N. "
             "Applies to both --seeds and --seeds-from-gt.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "closed_loop_cap" / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Subdir under output/_benchmark/. Default: timestamp.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (task, seed[, trial]) rollouts whose report.json already exists.",
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="repeats per seed. >1 emits seed_N/trial_KKK/; =1 keeps legacy seed_N/.",
    )
    parser.add_argument(
        "--start-trial", type=int, default=1,
        help="first trial index (1-indexed). Default 1 → trial_001.",
    )
    parser.add_argument(
        "--session", default=None,
        help="Session id for output (default: auto-generated YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--with-replay", action="store_true",
        help="After rollouts finish, run Phase 2 replay → LeRobot dataset.",
    )
    parser.add_argument(
        "--replay-include-failures", action="store_true",
        help="(with --with-replay) also replay failure episodes.",
    )
    parser.add_argument(
        "--replay-privileged", action="store_true",
        help="(with --with-replay) emit privileged oracle features.",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    if args.start_trial < 1:
        parser.error("--start-trial must be >= 1 (1-indexed)")

    logging.basicConfig(
        level=(
            logging.DEBUG if args.verbose >= 2
            else logging.INFO if args.verbose == 1
            else logging.WARNING
        ),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = _load_config(Path(args.config))

    # ---- Resolve seeds per task ----
    seeds_spec: list[int] | dict[str, list[int]]
    if args.seeds_from_gt:
        from closed_loop_cap.tools import load_or_find_gt_seeds

        collect = not args.no_auto_collect  # default: collect missing
        per_task: dict[str, list[int]] = {}
        for task in args.tasks:
            full = load_or_find_gt_seeds(
                task, config,
                max_attempts=args.gt_max_attempts,
                target_count=args.gt_target_count,
                rebuild=args.rebuild_gt_seeds,
                collect_if_missing=collect,
            )
            if not full and not collect:
                raise SystemExit(
                    f"[{task}] no cached GT seeds and --no-auto-collect set. "
                    f"Run `python -m closed_loop_cap.tools.find_gt_seeds --task {task}` first."
                )
            if args.num_seeds is not None:
                full = full[: args.num_seeds]
            per_task[task] = full
            logger.info("[%s] %d seeds selected from GT cache", task, len(full))
        seeds_spec = per_task
    else:
        base = args.seeds if args.seeds is not None else [0, 1, 2]
        if args.num_seeds is not None:
            base = base[: args.num_seeds]
        seeds_spec = base

    planner_client = _build_client(config, role="planner")
    codegen_client = _build_client(config, role="codegen")

    from closed_loop_cap.paths import default_session_id as _default_session_id
    session_id = args.session or _default_session_id()
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = args.output_suffix or ts
    bench_dir = (
        Path(REPO_ROOT) / config.get("output_dir", "closed_loop_cap/output") / "_benchmark" / suffix
    )

    summary = run_benchmark(
        tasks=args.tasks,
        seeds=seeds_spec,
        config=config,
        resume=args.resume,
        bench_dir=bench_dir,
        planner_client=planner_client,
        codegen_client=codegen_client,
        trials=args.trials,
        start_trial=args.start_trial,
        session=session_id,
    )
    summary["seeds_source"] = "gt_cache" if args.seeds_from_gt else "explicit"

    # Optional Phase 2: replay → LeRobot dataset, per task.
    if args.with_replay:
        from closed_loop_cap.dataset import DatasetRecorder, RecordingContext  # noqa: F401
        from closed_loop_cap.run_replay import (
            _camera_cfg, _iter_trials, _replay_one,
        )
        from closed_loop_cap.task_registry import load_task_meta as _load_meta
        from closed_loop_cap.paths import (
            logs_root as _paths_logs_root,
            recorded_data_dir,
        )
        replay_cfg = dict(config)
        replay_cfg.setdefault("logging", {})["privileged_features"] = bool(args.replay_privileged)
        for task in args.tasks:
            out_root = Path(REPO_ROOT) / replay_cfg.get("output_dir", "closed_loop_cap/output")
            logs_root = _paths_logs_root(out_root, session_id, task)
            pairs = _iter_trials(logs_root, None, None)
            if not pairs:
                logger.warning("[replay] no traj.pkl found for %s; skip", task)
                continue
            cameras_cfg, cam_h, cam_w = _camera_cfg(replay_cfg)
            task_meta = _load_meta(task)
            privileged_actors = (
                list(task_meta.actor_names) if args.replay_privileged else []
            )
            recorder = DatasetRecorder(
                root=recorded_data_dir(out_root, session_id, task),
                repo_id=replay_cfg.get("dataset", {})
                    .get("repo_id_template", "robotwin/{task}").format(task=task),
                fps=int(replay_cfg.get("logging", {}).get("fps", 30)),
                cameras=cameras_cfg,
                camera_height=cam_h,
                camera_width=cam_w,
                privileged_actors=privileged_actors,
            )
            try:
                for seed, trial in pairs:
                    try:
                        _replay_one(
                            task_name=task, seed=seed, trial=trial,
                            config=replay_cfg, recorder=recorder,
                            include_failures=args.replay_include_failures,
                            session=session_id,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "[replay] crash task=%s seed=%d trial=%d", task, seed, trial,
                        )
            finally:
                recorder.finalize()

    # Console summary
    print()
    print(format_markdown(summary))
    print(f"\nArtifacts: {bench_dir}")
    g = summary["grand_total"]
    return 0 if (g["valid"] > 0 and g["success"] == g["valid"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

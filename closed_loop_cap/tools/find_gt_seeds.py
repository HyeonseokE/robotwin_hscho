"""Pre-collect seeds for which the hand-written task solution (`envs/<task>.py`
`play_once()`) succeeds.

Why: closed_loop_cap episode failure has two very different causes:
    1. The VLM generated wrong code (the thing we want to measure).
    2. The seed's random object placement is simply not feasible for the
       built-in motion planner, so no scripted solution succeeds either.

Running the upstream GT solution once per seed separates these. We keep a
per-task cache of successful seeds under `output/_gt_seeds/<task>.json` so
repeated benchmarks reuse the same feasibility set.

CLI:
    python -m closed_loop_cap.tools.find_gt_seeds \
        --task beat_block_hammer \
        --max-attempts 50 --target-count 10

Programmatic:
    load_or_find_gt_seeds("beat_block_hammer", config) -> list[int]
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("closed_loop_cap.gt_seeds")


@dataclass
class GTSeedResult:
    task: str
    target_count: int
    max_attempts: int
    successful_seeds: list[int] = field(default_factory=list)
    failed_seeds: list[int] = field(default_factory=list)
    unstable_seeds: list[int] = field(default_factory=list)
    errored_seeds: list[dict] = field(default_factory=list)  # [{seed, error_type, msg}]
    total_attempts: int = 0
    duration_s: float = 0.0


# -----------------------------
# Path helpers
# -----------------------------


def _default_output_dir(config: dict) -> Path:
    return Path(REPO_ROOT) / config.get("output_dir", "closed_loop_cap/output") / "_gt_seeds"


def gt_seeds_path(task: str, config: dict) -> Path:
    return _default_output_dir(config) / f"{task}.json"


# -----------------------------
# Collection
# -----------------------------


def _run_single_seed(task_name: str, seed: int, args: dict) -> tuple[bool, str | None]:
    """Instantiate a fresh task, run play_once, return (success, failure_tag)."""
    module = importlib.import_module(f"envs.{task_name}")
    importlib.reload(module)
    task_cls = getattr(module, task_name)
    task = task_cls()
    try:
        task.setup_demo(now_ep_num=0, seed=seed, **args)
        task.play_once()
        ok = bool(getattr(task, "plan_success", True)) and bool(task.check_success())
        return ok, None if ok else "not_success"
    finally:
        try:
            task.close_env(clear_cache=True)
        except Exception:  # noqa: BLE001
            logger.debug("close_env failed for seed=%d", seed, exc_info=True)


def _is_unstable(exc: BaseException) -> bool:
    if type(exc).__name__ == "UnStableError":
        return True
    msg = str(exc).lower()
    return "unstable" in msg and "seed" in msg


def collect_gt_seeds(
    task_name: str,
    config: dict,
    *,
    max_attempts: int = 50,
    target_count: int = 10,
    start_seed: int = 0,
) -> GTSeedResult:
    """Iterate seeds until `target_count` successes or `max_attempts` budget."""
    # Lazy import — the env wrapper pulls in SAPIEN etc., which we don't want
    # at tools/__init__ import time (unit tests would break).
    from closed_loop_cap.env.task_env import _build_setup_args

    args = _build_setup_args(task_name, config)
    # Disable any video/data side-effects while scanning seeds.
    args["eval_video_save_dir"] = None
    args["collect_data"] = False
    args["save_data"] = False

    result = GTSeedResult(
        task=task_name, target_count=target_count, max_attempts=max_attempts,
    )
    t0 = time.monotonic()

    seed = start_seed
    while len(result.successful_seeds) < target_count and result.total_attempts < max_attempts:
        result.total_attempts += 1
        try:
            ok, _ = _run_single_seed(task_name, seed, args)
            if ok:
                result.successful_seeds.append(seed)
                logger.info(
                    "[%s] seed=%d OK (%d/%d)",
                    task_name, seed, len(result.successful_seeds), target_count,
                )
            else:
                result.failed_seeds.append(seed)
                logger.info("[%s] seed=%d not_success", task_name, seed)
        except Exception as exc:  # noqa: BLE001
            if _is_unstable(exc):
                result.unstable_seeds.append(seed)
                logger.info("[%s] seed=%d unstable", task_name, seed)
            else:
                result.errored_seeds.append(
                    {"seed": seed, "error_type": type(exc).__name__, "msg": str(exc)[:200]}
                )
                logger.warning("[%s] seed=%d ERROR %s: %s",
                               task_name, seed, type(exc).__name__, exc)
        seed += 1

    result.duration_s = round(time.monotonic() - t0, 2)
    return result


def save_gt_seeds(result: GTSeedResult, config: dict) -> Path:
    path = gt_seeds_path(result.task, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2))
    return path


def load_gt_seeds(task: str, config: dict) -> GTSeedResult | None:
    path = gt_seeds_path(task, config)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        return GTSeedResult(**{k: v for k, v in data.items() if k in GTSeedResult.__dataclass_fields__})
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load cached GT seeds for %s: %s", task, exc)
        return None


def load_or_find_gt_seeds(
    task_name: str,
    config: dict,
    *,
    max_attempts: int = 50,
    target_count: int = 10,
    rebuild: bool = False,
    collect_if_missing: bool = True,
) -> list[int]:
    """Return successful seeds for `task_name`. Uses cached json when available;
    runs the collector (and writes the cache) when missing or `rebuild=True`."""
    cached = None if rebuild else load_gt_seeds(task_name, config)
    if cached is not None and len(cached.successful_seeds) >= target_count:
        logger.info("[%s] using %d cached GT seeds", task_name, len(cached.successful_seeds))
        return list(cached.successful_seeds)

    if not collect_if_missing:
        return list(cached.successful_seeds) if cached else []

    logger.info("[%s] collecting GT seeds (target=%d, max=%d)",
                task_name, target_count, max_attempts)
    result = collect_gt_seeds(
        task_name, config, max_attempts=max_attempts, target_count=target_count,
    )
    save_gt_seeds(result, config)
    return list(result.successful_seeds)


# -----------------------------
# CLI
# -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect seeds whose hand-written play_once() succeeds.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "closed_loop_cap" / "configs" / "default.yaml"),
    )
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore any cached result for this task and re-collect.",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=(
            logging.DEBUG if args.verbose >= 2
            else logging.INFO if args.verbose == 1
            else logging.WARNING
        ),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cached = None if args.rebuild else load_gt_seeds(args.task, config)
    if cached is not None and len(cached.successful_seeds) >= args.target_count:
        print(f"[{args.task}] using cached result ({len(cached.successful_seeds)} seeds)")
        print(f"successful: {cached.successful_seeds[:args.target_count]}")
        return 0

    result = collect_gt_seeds(
        args.task, config,
        max_attempts=args.max_attempts,
        target_count=args.target_count,
        start_seed=args.start_seed,
    )
    path = save_gt_seeds(result, config)

    print(f"\nTask: {args.task}")
    print(f"Successful seeds ({len(result.successful_seeds)}/{result.target_count}): "
          f"{result.successful_seeds}")
    print(f"Failed seeds: {len(result.failed_seeds)}")
    print(f"Unstable seeds: {len(result.unstable_seeds)}")
    print(f"Errored: {len(result.errored_seeds)}")
    print(f"Total attempts: {result.total_attempts}, duration: {result.duration_s}s")
    print(f"Saved to: {path}")
    return 0 if len(result.successful_seeds) >= result.target_count else 1


if __name__ == "__main__":
    raise SystemExit(main())

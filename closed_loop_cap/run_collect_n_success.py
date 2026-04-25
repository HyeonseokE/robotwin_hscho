"""Collect exactly N successful trajectories for a single (task, seed).

Runs Phase 1 trials one at a time, skipping failures, until N successes
accumulate in the active session directory. Then runs Phase 2 replay on
successful trials only.

Usage:
    # Fresh session (auto-timestamped)
    python -m closed_loop_cap.run_collect_n_success \
        --task beat_block_hammer --seed 12 --target 10

    # Named session (append into existing directory)
    python -m closed_loop_cap.run_collect_n_success \
        --task beat_block_hammer --seed 12 --target 10 \
        --session perturb_off
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.paths import (  # noqa: E402
    default_session_id,
    seed_logs_dir,
)
from closed_loop_cap.run_closed_loop import (  # noqa: E402
    _build_client,
    _load_config,
    run_episode,
)

logger = logging.getLogger("closed_loop_cap.collect")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect N successful trajs for one seed")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target", type=int, default=10,
                        help="number of successful trajs to collect")
    parser.add_argument("--max-attempts", type=int, default=30,
                        help="give up after this many total trials")
    parser.add_argument("--session", default=None,
                        help=("Session id for output directory. Default: "
                              "auto-generated YYYYMMDD_HHMMSS."))
    parser.add_argument("--config", default=str(
        REPO_ROOT / "closed_loop_cap" / "configs" / "default.yaml"))
    parser.add_argument(
        "--recording-config", default=None,
        help=("Optional recording_config.yaml to overlay camera / feature / "
              "perturbation settings onto the main config."),
    )
    parser.add_argument("--with-replay", action="store_true",
                        help="run Phase 2 replay after collection")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose >= 2
               else logging.INFO if args.verbose == 1 else logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = _load_config(Path(args.config))
    if args.recording_config:
        from closed_loop_cap.configs.recording_config_loader import load_and_merge
        config = load_and_merge(config, args.recording_config)
    planner_client = _build_client(config, role="planner")
    codegen_client = _build_client(config, role="codegen")
    session_id = args.session or default_session_id()
    output_dir = REPO_ROOT / config.get("output_dir", "closed_loop_cap/output")

    successes = 0
    failures = 0
    trial = 1
    t0 = time.monotonic()

    print(f"Session: {session_id}")
    print(f"Collecting {args.target} successful trajs for {args.task} seed={args.seed}")
    print(f"Max attempts: {args.max_attempts}\n")

    while successes < args.target and trial <= args.max_attempts:
        print(f"--- trial {trial} (success {successes}/{args.target}, fail {failures}) ---")
        try:
            report = run_episode(
                args.task, seed=args.seed, config=config,
                planner_client=planner_client, codegen_client=codegen_client,
                trial=trial, session=session_id,
            )
            if report.success:
                successes += 1
                print(f"  SUCCESS (subtasks={report.num_subtasks})")
            else:
                failures += 1
                print(f"  FAIL (abort={report.abort_reason or '-'})")
        except Exception as exc:
            failures += 1
            print(f"  CRASH ({type(exc).__name__}: {exc})")
        trial += 1

    elapsed = time.monotonic() - t0
    print(f"\n{'='*50}")
    print(f"Session: {session_id}")
    print(f"Done: {successes} success, {failures} fail, {trial - 1} total attempts")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    if successes == 0:
        print("No successes — nothing to replay.")
        return 1

    # Phase 2: replay successful trials only
    if args.with_replay:
        print(f"\nStarting Phase 2 replay on {successes} successful trials...")
        from closed_loop_cap.run_replay import main as replay_main
        logs_root = seed_logs_dir(output_dir, session_id, args.task, args.seed)

        successful_trials = []
        for td in sorted(logs_root.iterdir()):
            if not td.is_dir():
                continue
            rp = td / "report.json"
            if rp.is_file():
                try:
                    rdata = json.loads(rp.read_text())
                    if rdata.get("success"):
                        successful_trials.append(int(td.name.removeprefix("trial_")))
                except Exception:
                    pass

        if successful_trials:
            replay_args = [
                "--task", args.task,
                "--seeds", str(args.seed),
                "--trials", *[str(t) for t in successful_trials],
                "--session", session_id,
                "--config", args.config,
            ]
            if args.verbose:
                replay_args.append("-v")
            sys.argv = ["run_replay"] + replay_args
            try:
                replay_main()
            except SystemExit:
                pass
            print(f"\nPhase 2 replay complete for {len(successful_trials)} episodes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

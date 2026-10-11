"""List the episodes of a LeRobot dataset that belong to one task.

Used to train the comparison policy on LIBERO's own robot demos of the same task
(lerobot/libero contains all 40 LIBERO tasks; we want only the ketchup one).
Only the dataset's metadata is downloaded, not the videos.

Usage:
    python src/reference_episodes.py                     # lerobot/libero, ketchup task
    python src/reference_episodes.py --limit 50          # cap the number of episodes
Prints a value to paste into lerobot-train as --dataset.episodes=...
"""

import argparse

TASK = "pick up the ketchup and place it in the basket"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="lerobot/libero")
    p.add_argument("--root", default=None, help="local copy of the dataset, if any")
    p.add_argument("--task", default=TASK)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    wanted = args.task.strip().lower()
    eps = [int(e["episode_index"]) for e in meta.episodes
           if any(t.strip().lower() == wanted for t in e["tasks"])]
    if not eps:
        known = sorted({t for e in meta.episodes for t in e["tasks"]})
        raise SystemExit(f"no episodes for task {args.task!r}. Tasks in this dataset:\n  " + "\n  ".join(known))
    if args.limit:
        eps = eps[: args.limit]
    frames = sum(int(meta.episodes[i]["length"]) for i in eps)
    print(f"{len(eps)} episodes, {frames} frames for task: {args.task}")
    print("--dataset.episodes=[" + ",".join(map(str, eps)) + "]")


if __name__ == "__main__":
    main()

"""List the episodes of a LeRobot dataset that belong to one task.

Used to train the comparison policy on LIBERO's own robot demos of the same task
(lerobot/libero contains all 40 LIBERO tasks; only the ketchup one is needed).

Some datasets list each episode's task in their metadata; others (such as
lerobot/libero) only store it per frame. In that case the small per-frame table
(states/actions, no videos) is downloaded and episodes are matched from it.

Usage:
    python src/reference_episodes.py                     # lerobot/libero, ketchup task
    python src/reference_episodes.py --limit 50          # cap the number of episodes
Prints a value to paste into lerobot-train as --dataset.episodes=...
"""

import argparse
from collections import defaultdict
from pathlib import Path

TASK = "pick up the ketchup and place it in the basket"


def norm(s):
    return " ".join(str(s).strip().lower().split())


def tasks_from_metadata(meta):
    """{episode_index: [task, ...]} if the episode metadata has a 'tasks' column, else None."""
    if meta.episodes is None or "tasks" not in meta.episodes.column_names:
        return None
    return {int(e["episode_index"]): list(e["tasks"]) for e in meta.episodes}


def tasks_from_frames(meta, repo_id, root):
    """{episode_index: [task, ...]} from the per-frame episode_index/task_index columns."""
    import pyarrow.parquet as pq

    root = Path(meta.root)
    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        from huggingface_hub import snapshot_download

        print("episode metadata has no task names; downloading the per-frame table (no videos)...")
        snapshot_download(repo_id=repo_id, repo_type="dataset", revision=meta.revision,
                          allow_patterns=["data/*", "data/**/*"], local_dir=str(root))
        files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        raise SystemExit(f"no per-frame data found under {root}/data")

    if "task_index" in meta.tasks.columns:
        index_to_task = {int(i): t for t, i in meta.tasks["task_index"].items()}
    else:  # tasks table indexed by task_index with a 'task' column
        index_to_task = {int(i): t for i, t in meta.tasks["task"].items()}

    by_episode = defaultdict(set)
    for f in files:
        tab = pq.read_table(f, columns=["episode_index", "task_index"]).to_pandas()
        for (ep, ti), _ in tab.groupby(["episode_index", "task_index"]):
            by_episode[int(ep)].add(index_to_task.get(int(ti), f"<task {ti}>"))
    return {ep: sorted(ts) for ep, ts in by_episode.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="lerobot/libero")
    p.add_argument("--root", default=None, help="local copy of the dataset, if any")
    p.add_argument("--task", default=TASK)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    ep_tasks = tasks_from_metadata(meta)
    if ep_tasks is None:
        ep_tasks = tasks_from_frames(meta, args.repo_id, args.root)

    wanted = norm(args.task)
    eps = sorted(ep for ep, ts in ep_tasks.items() if any(norm(t) == wanted for t in ts))
    if not eps:
        known = sorted({t for ts in ep_tasks.values() for t in ts})
        raise SystemExit(f"no episodes for task {args.task!r}. Tasks in this dataset:\n  " + "\n  ".join(known))
    if args.limit:
        eps = eps[: args.limit]
    lengths = {int(e["episode_index"]): int(e["length"]) for e in meta.episodes}
    frames = sum(lengths.get(i, 0) for i in eps)
    print(f"{len(eps)} episodes, {frames} frames for task: {args.task}")
    print("--dataset.episodes=[" + ",".join(map(str, eps)) + "]")


if __name__ == "__main__":
    main()

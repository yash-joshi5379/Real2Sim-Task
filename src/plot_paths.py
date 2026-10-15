"""Plot the robot's gripper paths in the training data of runs D and G (docs/media/paths_D_vs_G.png).

Replays all 54 demos twice without cameras (each demo's own path, and the averaged
path shape) and draws the gripper positions, top view and side view. ~10 min on a CPU.
Usage:  python src/plot_paths.py
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_dataset as M  # noqa: E402
import replay_in_sim as R  # noqa: E402


def gripper_paths(traj_dir):
    names = sorted(f.stem for f in traj_dir.glob("ep_*.npz"))
    trajs = {n: dict(np.load(traj_dir / f"{n}.npz")) for n in names}
    metas = {n: json.loads((traj_dir / f"{n}.json").read_text()) for n in names}
    T = R.fit_xy_transform(list(trajs.values()))
    env, inits, _ = R.make_env(64, cameras=False)
    jobs = []
    for k, n in enumerate(names):                        # same as --placement eval --init-state all
        home = R.reset_with_ketchup(env, inits[k % 50], R.KETCHUP_SPAWN)["robot0_eef_pos"].copy()
        basket = env.env.sim.data.get_joint_qpos("basket_1_joint0")[:2].copy()
        plan, info = R.retarget(trajs[n], metas[n], T, home, basket, 1.3)
        jobs.append((k % 50, *M.shift_plan(plan, info, R.KETCHUP_SPAWN - np.array(info["ketchup_xy"]))))
    averaged = copy.deepcopy([j[1] for j in jobs])
    M.blend_paths(averaged, 1.0)
    out = {}
    for tag, plans in (("D", [j[1] for j in jobs]), ("G", averaged)):
        out[tag] = []
        for (li, _, info), plan in zip(jobs, plans):
            res = R.run_episode(env, inits[li], plan, info)
            end = res["success_step"] + 11 if res["success"] else len(res["states"])
            out[tag].append(res["states"][:end, :3])     # what the training episode contains
    return out


def plot(paths, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, ink, ink2, grid, bg = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), dpi=150, sharex=True)
    fig.patch.set_facecolor(bg)
    titles = {"D": "D: my own path in every demo  (50%)", "G": "G: average path shape  (96%)"}
    for c, tag in enumerate(("D", "G")):
        top, side = axes[0, c], axes[1, c]
        for p in paths[tag]:
            top.plot(p[:, 1] * 100, p[:, 0] * 100, color=blue, alpha=0.35, lw=1.1)
            side.plot(p[:, 1] * 100, p[:, 2] * 100, color=blue, alpha=0.35, lw=1.1)
        top.set_title(titles[tag], loc="left", fontsize=12, fontweight="bold", color=ink)
        for ax in (top, side):
            ax.set_facecolor(bg)
            ax.grid(True, color=grid, lw=0.7)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(grid)
            ax.tick_params(colors=ink2, length=0)
        top.scatter([-24], [-12], s=70, color=ink, zorder=5)
        top.annotate("ketchup", (-24, -12), (-24, -3.5), ha="center", color=ink, fontsize=10)
        top.scatter([26], [0], s=70, marker="s", color=ink, zorder=5)
        top.annotate("basket", (26, 0), (26, 7.5), ha="center", color=ink, fontsize=10)
        side.axhline(R.BASKET_RIM * 100, color=ink2, lw=1, ls="--")
        side.text(31.5, 14.6, "basket rim", ha="right", color=ink2, fontsize=9.5)
        side.set_xlabel("from the ketchup towards the basket (cm)", color=ink2)
    axes[0, 0].set_ylabel("top view: sideways (cm)", color=ink2)
    axes[1, 0].set_ylabel("side view: height (cm)", color=ink2)
    for r in range(2):
        lo = min(axes[r, 0].get_ylim()[0], axes[r, 1].get_ylim()[0])
        hi = max(axes[r, 0].get_ylim()[1], axes[r, 1].get_ylim()[1])
        for c in range(2):
            axes[r, c].set_ylim(lo, hi)
    fig.suptitle("Robot gripper paths in the training data: all 54 demos from my phone videos",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=bg)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    plot(gripper_paths(root / "data/trajectories"), root / "docs/media/paths_D_vs_G.png")

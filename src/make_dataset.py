"""Turn the retargeted replays into a LeRobot dataset for training (SmolVLA or ACT).

Each demo is replayed in LIBERO as in replay_in_sim.py, this time with both cameras
on, and every control step is recorded. Only replays that put the ketchup in the
basket are kept, and each episode is cut 10 steps after success (evaluation stops at
success, so a policy never needs what comes after).

Same format as LeRobot's own LIBERO data (lerobot/libero) and what lerobot-eval feeds
a policy, so one policy config trains on both:
  observation.images.image    agentview camera, 256x256, rotated 180 deg
  observation.images.image2   wrist camera,     256x256, rotated 180 deg
  observation.state           eef xyz, eef axis-angle (3), gripper finger positions (2)
  action                      7-D delta end-effector + gripper, in [-1, 1]
  task                        "pick up the ketchup and place it in the basket"

Options (each experiment in the README is one combination, see setup/run_experiment.sh):
  --placement  human: ketchup where the hand found the bottle in the video
               spawn: random points in LIBERO's 5x5 cm ketchup spawn region
               eval:  exactly where all 50 evaluation layouts put it (-0.12, -0.24)
  --init-state a layout number (default 0), or "all" to cycle through LIBERO's 50 layouts
  --motion     human: the hand's reach and carry (default)
               timed: the hand's paths, but one fixed smooth timing for every demo
               scripted: control, straight minimum-jerk paths instead of the hand's
  --blend A    pull each path's shape towards the average shape of all demos (0..1)
  --smooth S   Savitzky-Golay filter over S seconds, to remove hand-tracking jitter
  --augment K  K extra replays per demo (with --placement human: ketchup nudged <= 3 cm)
  --dry-run    replay everything without cameras, report success, write nothing

Example (run G, the best one):
    python src/make_dataset.py --placement eval --init-state all --blend 1.0 \\
        --out data/lerobot/ketchup_phone_blend10 --repo-id local/ketchup_phone_blend10
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_in_sim as R  # noqa: E402

IMG = 256
TAIL_AFTER_SUCCESS = 10
SPAWN_LO = np.array([-0.145, -0.265])   # LIBERO's ketchup spawn region (task BDDL file)
SPAWN_HI = np.array([-0.095, -0.215])

FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (IMG, IMG, 3),
                                 "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "video", "shape": (IMG, IMG, 3),
                                  "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,),
                          "names": ["x", "y", "z", "ax", "ay", "az", "finger_l", "finger_r"]},
    "action": {"dtype": "float32", "shape": (7,),
               "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
}


def copy_plan(plan):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in plan.items()}


def minjerk_profile(n):
    u = np.linspace(0, 1, max(n, 2))
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def shift_plan(plan, info, d):
    """Move the ketchup by d (xy). The reach is bent to end at the new grasp point; the
    carry starts there and blends back onto the original path within 40% of its length."""
    plan, info = copy_plan(plan), dict(info)
    d3 = np.array([d[0], d[1], 0.0])
    info["ketchup_xy"] = (np.array(info["ketchup_xy"]) + d).tolist()
    plan["grasp"] = plan["grasp"] + d3
    plan["pre_grasp"] = plan["pre_grasp"] + d3
    plan["reach"] = plan["reach"] + R.smoothstep(np.linspace(0, 1, len(plan["reach"])))[:, None] * d3
    plan["carry"] = plan["carry"] + (1 - R.smoothstep(np.linspace(0, 1, len(plan["carry"])) / 0.4))[:, None] * d3
    return plan, info


def scripted_plan(plan, home, reach_steps, carry_steps):
    """Control: replace the hand's reach and carry with straight minimum-jerk moves.

    Everything else stays the same (ketchup position, grasp height, release point,
    robot-side steps). reach: start -> above the ketchup. carry: up to the same peak
    height, across, down to the same release (30 / 45 / 25% of the time). Durations
    are the median over all demos.
    """
    plan = copy_plan(plan)
    plan["reach"] = home + (plan["pre_grasp"] - home) * minjerk_profile(reach_steps)[:, None]
    g, rel = plan["grasp"], plan["carry"][-1]
    peak_z = max(plan["carry"][:, 2].max(), plan["rim_z"])
    up, across = np.r_[g[:2], peak_z], np.r_[rel[:2], peak_z]
    n1, n2 = int(0.30 * carry_steps), int(0.45 * carry_steps)
    n3 = carry_steps - n1 - n2
    plan["carry"] = np.concatenate([g + (up - g) * minjerk_profile(n1)[:, None],
                                    up + (across - up) * minjerk_profile(n2)[:, None],
                                    across + (rel - across) * minjerk_profile(n3)[:, None]])
    return plan


def retime(path, n):
    """Same path shape, new timing: n steps along the path with a minimum-jerk speed profile."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if s[-1] < 1e-6:
        return np.repeat(path[:1], max(n, 2), axis=0)
    target = s[-1] * minjerk_profile(n)
    return np.stack([np.interp(target, s, path[:, k]) for k in range(3)], axis=1)


def timed_plan(plan, reach_steps, carry_steps):
    plan = copy_plan(plan)
    plan["reach"] = retime(plan["reach"], reach_steps)
    plan["carry"] = retime(plan["carry"], carry_steps)
    return plan


def resample(path, n):
    """A path re-sampled to n points evenly spaced in time."""
    t, u = np.linspace(0, 1, len(path)), np.linspace(0, 1, max(n, 2))
    return np.stack([np.interp(u, t, path[:, k]) for k in range(3)], axis=1)


def blend_paths(plans, alpha, m=200):
    """Pull every reach and carry towards the average shape of all of them (in place).

    Each path = the straight line from its start to its end + its deviation from that
    line. The deviations are averaged over normalised time (so the average is still made
    of the recorded hand motion) and mixed back into every path. Start and end points are
    unchanged, so every demo still starts at home and grasps/releases where it did, and
    keeps its own duration.
    """
    for key in ("reach", "carry"):
        dev = []
        for pl in plans:
            q = resample(pl[key], m)
            dev.append(q - (q[0] + np.linspace(0, 1, m)[:, None] * (q[-1] - q[0])))
        mean_dev = np.mean(dev, axis=0)
        for pl in plans:
            a, b, n = pl[key][0], pl[key][-1], len(pl[key])
            line = a + np.linspace(0, 1, n)[:, None] * (b - a)
            pl[key] = line + (1 - alpha) * (pl[key] - line) + alpha * resample(mean_dev, n)


def smooth_paths(plan, seconds):
    """Savitzky-Golay filter on the reach and carry; start and end points are kept exactly."""
    from scipy.signal import savgol_filter

    plan = copy_plan(plan)
    w = int(seconds * R.CTRL_HZ) | 1                     # odd window length
    for key in ("reach", "carry"):
        x = plan[key]
        if len(x) <= w or w < 5:
            continue
        sm = savgol_filter(x, w, 2, axis=0)
        ramp = np.linspace(0, 1, len(x))[:, None]
        plan[key] = sm + (1 - ramp) * (x[0] - sm[0]) + ramp * (x[-1] - sm[-1])
    return plan


def record_episode(env, init_state, plan, info):
    """Replay with cameras on; return the recorded steps, or None if the replay failed."""
    from robosuite.utils import transform_utils as tu

    steps = []

    def hook(obs, action, phase):
        steps.append({
            "observation.images.image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            "observation.images.image2": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
            "observation.state": np.r_[obs["robot0_eef_pos"], tu.quat2axisangle(obs["robot0_eef_quat"]),
                                       obs["robot0_gripper_qpos"]].astype(np.float32),
            "action": np.asarray(action, dtype=np.float32),
        })

    res = R.run_episode(env, init_state, plan, info, hook=hook)
    if not res["success"] or res["success_step"] is None:
        return None, res
    return steps[: res["success_step"] + 1 + TAIL_AFTER_SUCCESS], res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", default="data/trajectories")
    p.add_argument("--out", default="data/lerobot/ketchup_phone")
    p.add_argument("--repo-id", default="local/ketchup_phone")
    p.add_argument("--episodes", nargs="+", help="e.g. ep_001 ep_002 (default: all)")
    p.add_argument("--speed", type=float, default=1.3, help="play the human motion this many times faster")
    p.add_argument("--placement", choices=["human", "spawn", "eval"], default="human")
    p.add_argument("--init-state", default="0", help='layout number, or "all" to cycle through the 50')
    p.add_argument("--motion", choices=["human", "timed", "scripted"], default="human")
    p.add_argument("--blend", type=float, default=0.0)
    p.add_argument("--smooth", type=float, default=0.0)
    p.add_argument("--augment", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    if not args.dry_run and out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    traj_dir = Path(args.traj)
    names = sorted(f.stem for f in traj_dir.glob("ep_*.npz"))
    if not names:
        sys.exit(f"no ep_*.npz trajectories in {traj_dir}")
    trajs = {n: dict(np.load(traj_dir / f"{n}.npz")) for n in names}
    metas = {n: json.loads((traj_dir / f"{n}.json").read_text()) for n in names}
    T = R.fit_xy_transform(list(trajs.values()))          # always fitted on all demos
    if args.episodes:
        names = [n for n in names if n in args.episodes]

    env, inits, language = R.make_env(IMG, cameras=not args.dry_run)
    layouts = {}

    def layout(i):
        """Robot start position and basket position in LIBERO layout i."""
        if i not in layouts:
            home = R.reset_with_ketchup(env, inits[i], R.KETCHUP_SPAWN)["robot0_eef_pos"].copy()
            layouts[i] = (home, env.env.sim.data.get_joint_qpos("basket_1_joint0")[:2].copy())
        return layouts[i]

    rng = np.random.default_rng(args.seed)
    if args.motion != "human":                  # fixed durations: the median over all demos
        home0, basket0 = layout(0)
        plans0 = [R.retarget(trajs[n], metas[n], T, home0, basket0, args.speed)[0] for n in trajs]
        med_reach = int(np.median([len(pl["reach"]) for pl in plans0]))
        med_carry = int(np.median([len(pl["carry"]) for pl in plans0]))
        print(f"{args.motion} motion: reach {med_reach} steps, carry {med_carry} steps")

    # pass 1: every episode's plan (blending needs all of them at once)
    jobs, count = [], 0
    for n in names:
        for v in range(1 + args.augment):
            li = count % len(inits) if args.init_state == "all" else int(args.init_state)
            count += 1
            home, basket_xy = layout(li)
            plan, info = R.retarget(trajs[n], metas[n], T, home, basket_xy, args.speed)
            if args.motion == "scripted":
                plan = scripted_plan(plan, home, med_reach, med_carry)
            elif args.motion == "timed":
                plan = timed_plan(plan, med_reach, med_carry)
            if args.placement == "human":
                d = np.zeros(2) if v == 0 else rng.uniform(-0.03, 0.03, size=2)
            elif args.placement == "spawn":
                d = rng.uniform(SPAWN_LO, SPAWN_HI) - np.array(info["ketchup_xy"])
            else:
                d = R.KETCHUP_SPAWN - np.array(info["ketchup_xy"])
            if args.smooth > 0:
                plan = smooth_paths(plan, args.smooth)
            if np.any(d != 0):
                plan, info = shift_plan(plan, info, d)
            jobs.append((n, v, li, plan, info, d))
    if args.blend > 0:
        blend_paths([j[3] for j in jobs], args.blend)
        print(f"paths blended {args.blend:.0%} towards the average shape")

    ds = None
    if not args.dry_run:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset.create(repo_id=args.repo_id, fps=R.CTRL_HZ, features=FEATURES, root=out,
                                   robot_type="panda", use_videos=True)

    # pass 2: replay and record
    index, skipped = [], []
    for n, v, li, plan, info, d in jobs:
        tag = n if v == 0 else f"{n}+aug{v}"
        if args.dry_run:
            res = R.run_episode(env, inits[li], plan, info)
            steps = [None] * (res["success_step"] + 1 + TAIL_AFTER_SUCCESS) if res["success"] else None
        else:
            steps, res = record_episode(env, inits[li], plan, info)
        if steps is None:
            skipped.append(tag)
            print(f"{tag}: replay failed, skipped")
            continue
        if ds is not None:
            for frame in steps:
                ds.add_frame({**frame, "task": language})
            ds.save_episode()
        index.append({"episode_index": len(index), "source_demo": n, "variant": v, "layout": li,
                      "ketchup_shift_m": [round(float(x), 4) for x in d], "frames": len(steps),
                      "success_step": res["success_step"]})
        print(f"{tag}: layout {li}, {len(steps)} frames (success at step {res['success_step']})")

    frames = sum(e["frames"] for e in index)
    if args.dry_run:
        print(f"\ndry run: {len(index)}/{len(jobs)} replays succeed, {frames} frames; nothing written")
        return
    ds.finalize()
    settings = {k: getattr(args, k) for k in ("speed", "placement", "init_state", "motion", "blend",
                                               "smooth", "augment", "seed")}
    (out / "episode_sources.json").write_text(json.dumps(
        {**settings, "transform": T, "skipped": skipped, "episodes": index}, indent=2))
    print(f"\n{len(index)} episodes, {frames} frames -> {out}  (skipped {len(skipped)})")


if __name__ == "__main__":
    main()

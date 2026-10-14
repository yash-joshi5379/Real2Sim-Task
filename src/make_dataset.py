"""Turn the retargeted replays into a LeRobot dataset for training (e.g. SmolVLA).

Each demo is replayed in LIBERO exactly as in replay_in_sim.py (deterministic), this
time with both cameras on, and every control step is recorded. Only replays that
put the ketchup in the basket are kept, and each episode ends shortly after
success (LIBERO evaluation stops at success, so a policy never sees what comes after).

The format matches LeRobot's own LIBERO datasets (lerobot/libero) and what
lerobot-eval feeds a policy, so the same policy config trains on both:
  observation.images.image    agentview camera, 256x256, rotated 180 deg
  observation.images.image2   wrist camera,     256x256, rotated 180 deg
  observation.state           [eef xyz, eef axis-angle (3), gripper finger qpos (2)]
  action                      7-D delta end-effector + gripper, in [-1, 1]
  task                        "pick up the ketchup and place it in the basket"

Motion (--motion): human (default) uses your reach and carry; timed keeps your paths'
shapes but gives every demo the same smooth timing (median durations); scripted is a
control experiment that swaps them for smooth standard-shape paths, keeping everything else.

Ketchup placement:
  --placement human  (default) the ketchup starts where your hand found the bottle;
                     --augment K adds K copies nudged by up to 3 cm.
  --placement spawn  the ketchup starts at random points inside LIBERO's own spawn
                     region (a 5x5 cm area, where evaluation puts it), 1 + K copies per
                     demo. Same human motion, bent to meet the ketchup.
  --placement eval   the ketchup starts exactly where all 50 evaluation layouts put it
                     (-0.12, -0.24); LIBERO's own demos (run B) also all have it there.

Starting layout (--init-state): a number uses that one LIBERO layout for every episode
(default 0); "all" cycles through the 50 layouts (robot start pose differs by ~1 cm,
basket by ~1 cm), as LIBERO's own demos do.

--dry-run replays every episode without cameras and reports success, writing nothing.

Usage:
    python src/make_dataset.py --speed 1.3                    # 54 episodes
    python src/make_dataset.py --speed 1.3 --placement spawn --augment 3 \\
        --out data/lerobot/ketchup_from_phone_spawn4 --repo-id local/ketchup_from_phone_spawn4
    python src/make_dataset.py --placement eval --init-state all --dry-run
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
TAIL_AFTER_SUCCESS = 10            # steps kept after the ketchup lands (gripper opening)
# LIBERO's spawn region for the ketchup (target_object_region in the task's BDDL file), xy in m
SPAWN_LO = np.array([-0.145, -0.265])
SPAWN_HI = np.array([-0.095, -0.215])
# where the ketchup actually is in all 50 evaluation layouts (measured; identical in each)
EVAL_SPOT = np.array([-0.12, -0.24])

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


def shift_plan(plan, info, d):
    """Move the ketchup by d (xy) and bend the path to meet it.

    The reach ends at the new grasp point; the carry starts there and blends back
    onto your original carry path by 40% of the way (the release is unchanged).
    """
    plan = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in plan.items()}
    info = dict(info)
    d3 = np.array([d[0], d[1], 0.0])
    info["ketchup_xy"] = (np.array(info["ketchup_xy"]) + d).tolist()
    plan["grasp"] = plan["grasp"] + d3
    plan["pre_grasp"] = plan["pre_grasp"] + d3
    n = len(plan["reach"])
    plan["reach"] = plan["reach"] + R.smoothstep(np.linspace(0, 1, n))[:, None] * d3
    n = len(plan["carry"])
    plan["carry"] = plan["carry"] + (1 - R.smoothstep(np.linspace(0, 1, n) / 0.4))[:, None] * d3
    return plan, info


def scripted_plan(plan, info, home, reach_steps, carry_steps):
    """Control experiment: replace your reach and carry with smooth, standard-shape paths.

    Everything else stays as in the human version (ketchup position, grasp height,
    release point, robot-side waits), so comparing the two isolates the effect of the
    human motion itself (its shape, wobble and timing).
      reach: straight, minimum-jerk move from the start pose to above the ketchup
      carry: lift straight up to the same peak height, move across, lower to the
             same release point (30% / 45% / 25% of the time)
    Durations are fixed (the median over your demos), not per demo.
    """
    plan = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in plan.items()}

    def minjerk(a, b, n):
        u = np.linspace(0, 1, max(n, 2))
        w = 10 * u**3 - 15 * u**4 + 6 * u**5
        return a + (b - a) * w[:, None]

    plan["reach"] = minjerk(home, plan["pre_grasp"], reach_steps)
    g, rel = plan["grasp"], plan["carry"][-1]
    peak_z = max(plan["carry"][:, 2].max(), plan["rim_z"])
    up, across = np.r_[g[:2], peak_z], np.r_[rel[:2], peak_z]
    n1, n2 = int(0.30 * carry_steps), int(0.45 * carry_steps)
    plan["carry"] = np.concatenate([minjerk(g, up, n1), minjerk(up, across, n2),
                                    minjerk(across, rel, carry_steps - n1 - n2)])
    return plan, dict(info)


def retime(path, n):
    """Same path shape, new timing: n steps with a smooth (minimum-jerk) speed profile.

    Positions are re-sampled by distance travelled along your path, so pauses and
    speed changes disappear but every bend and height change of the path is kept.
    """
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-6:
        return np.repeat(path[:1], max(n, 2), axis=0)
    u = np.linspace(0, 1, max(n, 2))
    target = s[-1] * (10 * u**3 - 15 * u**4 + 6 * u**5)
    return np.stack([np.interp(target, s, path[:, k]) for k in range(3)], axis=1)


def timed_plan(plan, info, reach_steps, carry_steps):
    """Your reach and carry paths, played with the same fixed, smooth timing in every demo."""
    plan = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in plan.items()}
    plan["reach"] = retime(plan["reach"], reach_steps)
    plan["carry"] = retime(plan["carry"], carry_steps)
    return plan, dict(info)


def resample(path, n):
    """A path re-sampled to n points evenly spaced in time (keeps its own speed profile)."""
    t = np.linspace(0, 1, len(path))
    u = np.linspace(0, 1, max(n, 2))
    return np.stack([np.interp(u, t, path[:, k]) for k in range(3)], axis=1)


def blend_paths(plans, alpha, m=200):
    """Pull every reach and carry towards the average of all of them (in place).

    Each path is split into its straight line from start to end plus its sideways/vertical
    deviation from that line. The deviations are averaged over normalised time (the mean of
    your 54 path shapes, so still your motion) and mixed into each path. Starts and ends are
    unchanged, so the robot still begins at home and grasps/releases exactly where it did,
    and each demo keeps its own duration (its speed along the path is blended the same way).
    """
    for key in ("reach", "carry"):
        ends = np.array([[pl[key][0], pl[key][-1]] for pl in plans])
        # average shape relative to each path's own straight line from start to end
        dev = []
        for pl in plans:
            q = resample(pl[key], m)
            line = q[0] + np.linspace(0, 1, m)[:, None] * (q[-1] - q[0])
            dev.append(q - line)
        mean_dev = np.mean(dev, axis=0)
        for pl, (a, b) in zip(plans, ends):
            n = len(pl[key])
            line = a + np.linspace(0, 1, n)[:, None] * (b - a)
            own_dev = pl[key] - line
            pl[key] = line + (1 - alpha) * own_dev + alpha * resample(mean_dev, n)


def record_episode(env, init_state, plan, info):
    """Replay with cameras on; return per-step frames, or None if the replay failed."""
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

    res = R.run_episode(env, init_state, plan, info, record=False, hook=hook)
    if not res["success"] or res["success_step"] is None:
        return None, res
    return steps[: res["success_step"] + 1 + TAIL_AFTER_SUCCESS], res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", default="data/trajectories")
    p.add_argument("--out", default="data/lerobot/ketchup_from_phone")
    p.add_argument("--repo-id", default="local/ketchup_from_phone")
    p.add_argument("--episodes", nargs="+", help="e.g. ep_001 ep_002 (default: all)")
    p.add_argument("--speed", type=float, default=1.3)
    p.add_argument("--augment", type=int, default=0, help="extra replays per demo with the ketchup moved")
    p.add_argument("--motion", choices=["human", "timed", "scripted"], default="human",
                   help="timed: your paths with one fixed smooth timing; scripted: control experiment "
                        "with smooth standard reach/carry paths")
    p.add_argument("--blend", type=float, default=0.0,
                   help="0..1: pull each of your reach/carry paths towards the average of all your paths "
                        "(0 = your own path, 1 = the average path, keeping each demo's own timing)")
    p.add_argument("--placement", choices=["human", "spawn", "eval"], default="human",
                   help="human: ketchup where your hand found the bottle; spawn: random points in "
                        "LIBERO's ketchup spawn region; eval: exactly where evaluation puts it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-state", default="0", help='layout number, or "all" to cycle through the 50')
    p.add_argument("--dry-run", action="store_true", help="replay without cameras, report success, write nothing")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    if not args.dry_run and out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    traj_dir = Path(args.traj)
    names = sorted(f.stem for f in traj_dir.glob("ep_*.npz"))
    trajs = {n: dict(np.load(traj_dir / f"{n}.npz")) for n in names}
    metas = {n: json.loads((traj_dir / f"{n}.json").read_text()) for n in names}
    T = R.fit_xy_transform(list(trajs.values()))          # fitted on all demos, as in replay_in_sim
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
    if args.motion in ("scripted", "timed"):   # fixed durations: the median of your demos at this speed
        home0, basket0 = layout(0)
        lens = [R.retarget(trajs[n], metas[n], T, home0, basket0, args.speed)[0] for n in trajs]
        med_reach = int(np.median([len(pl["reach"]) for pl in lens]))
        med_carry = int(np.median([len(pl["carry"]) for pl in lens]))
        print(f"{args.motion} motion: reach {med_reach} steps, carry {med_carry} steps")

    ds = None
    if not args.dry_run:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset.create(repo_id=args.repo_id, fps=R.CTRL_HZ, features=FEATURES, root=out,
                                   robot_type="panda", use_videos=True)
    # pass 1: build every episode's plan
    jobs, count = [], 0
    for n in names:
        for v in range(1 + args.augment):
            li = count % len(inits) if args.init_state == "all" else int(args.init_state)
            count += 1
            home, basket_xy = layout(li)
            plan, info = R.retarget(trajs[n], metas[n], T, home, basket_xy, args.speed)
            if args.motion == "scripted":
                plan, info = scripted_plan(plan, info, home, med_reach, med_carry)
            elif args.motion == "timed":
                plan, info = timed_plan(plan, info, med_reach, med_carry)
            if args.placement == "human":
                # where your hand found the bottle (+ optional random nudges of up to 3 cm)
                d = np.zeros(2) if v == 0 else rng.uniform(-0.03, 0.03, size=2)
            elif args.placement == "spawn":
                # random points in LIBERO's spawn region; your motion bends to meet the ketchup
                d = rng.uniform(SPAWN_LO, SPAWN_HI) - np.array(info["ketchup_xy"])
            else:
                # exactly where every evaluation layout (and every LIBERO demo) has the ketchup
                d = EVAL_SPOT - np.array(info["ketchup_xy"])
            if np.any(d != 0):
                plan, info = shift_plan(plan, info, d)
            jobs.append((n, v, li, plan, info, d))
    if args.blend > 0:
        blend_paths([j[3] for j in jobs], args.blend)
        print(f"blended every reach and carry {args.blend:.0%} towards the average of your paths")

    # pass 2: replay and record
    index, skipped = [], []
    for n, v, li, plan, info, d in jobs:
        tag = n if v == 0 else f"{n}+aug{v}"
        if args.dry_run:
            res = R.run_episode(env, inits[li], plan, info)
            steps = None if not res["success"] else [None] * (res["success_step"] + 1 + TAIL_AFTER_SUCCESS)
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
        print(f"\ndry run: {len(index)}/{len(index) + len(skipped)} replays succeed, {frames} frames "
              f"(longest {max(e['frames'] for e in index)}); nothing written")
        return
    ds.finalize()

    (out / "episode_sources.json").write_text(json.dumps(
        {"speed": args.speed, "augment": args.augment, "placement": args.placement, "motion": args.motion,
         "blend": args.blend,
         "seed": args.seed,
         "init_state": args.init_state,
         "transform": T, "skipped": skipped, "episodes": index}, indent=2))
    print(f"\n{len(index)} episodes, {frames} frames -> {out}  (skipped {len(skipped)})")


if __name__ == "__main__":
    main()

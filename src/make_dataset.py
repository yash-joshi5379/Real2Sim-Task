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

Ketchup placement:
  --placement human  (default) the ketchup starts where your hand found the bottle;
                     --augment K adds K copies nudged by up to 3 cm.
  --placement spawn  the ketchup starts at random points inside LIBERO's own spawn
                     region (a 5x5 cm area, where evaluation puts it), 1 + K copies per
                     demo. Same human motion, bent to meet the ketchup.

Usage:
    python src/make_dataset.py --speed 1.3                    # 54 episodes
    python src/make_dataset.py --speed 1.3 --placement spawn --augment 3 \\
        --out data/lerobot/ketchup_from_phone_spawn4 --repo-id local/ketchup_from_phone_spawn4
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
    p.add_argument("--placement", choices=["human", "spawn"], default="human",
                   help="human: ketchup where your hand found the bottle; spawn: random points in "
                        "LIBERO's ketchup spawn region (where evaluation puts it)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-state", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out)
    if out.exists():
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

    env, inits, language = R.make_env(IMG, cameras=True)
    init_state = inits[args.init_state]
    home = R.reset_with_ketchup(env, init_state, R.KETCHUP_SPAWN)["robot0_eef_pos"].copy()
    basket_xy = env.env.sim.data.get_joint_qpos("basket_1_joint0")[:2].copy()
    rng = np.random.default_rng(args.seed)

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=R.CTRL_HZ, features=FEATURES, root=out,
                               robot_type="panda", use_videos=True)
    index, skipped = [], []
    for n in names:
        plan0, info0 = R.retarget(trajs[n], metas[n], T, home, basket_xy, args.speed)
        variants = []
        if args.placement == "human":
            # where your hand found the bottle (+ optional random nudges of up to 3 cm)
            variants.append((plan0, info0, [0.0, 0.0]))
            for _ in range(args.augment):
                d = rng.uniform(-0.03, 0.03, size=2)
                variants.append((*shift_plan(plan0, info0, d), d.tolist()))
        else:
            # "spawn": the ketchup at random points inside LIBERO's own spawn region, the
            # region evaluation layouts use; your motion bends to meet it (1 + augment copies)
            for _ in range(1 + args.augment):
                target = rng.uniform(SPAWN_LO, SPAWN_HI)
                d = target - np.array(info0["ketchup_xy"])
                variants.append((*shift_plan(plan0, info0, d), d.tolist()))
        for v, (plan, info, d) in enumerate(variants):
            steps, res = record_episode(env, init_state, plan, info)
            tag = n if v == 0 else f"{n}+aug{v}"
            if steps is None:
                skipped.append(tag)
                print(f"{tag}: replay failed, skipped")
                continue
            for frame in steps:
                ds.add_frame({**frame, "task": language})
            ds.save_episode()
            index.append({"episode_index": len(index), "source_demo": n, "variant": v,
                          "ketchup_shift_m": [round(x, 4) for x in d], "frames": len(steps),
                          "success_step": res["success_step"]})
            print(f"{tag}: {len(steps)} frames (success at step {res['success_step']})")
    ds.finalize()

    (out / "episode_sources.json").write_text(json.dumps(
        {"speed": args.speed, "augment": args.augment, "placement": args.placement, "seed": args.seed,
         "init_state": args.init_state,
         "transform": T, "skipped": skipped, "episodes": index}, indent=2))
    frames = sum(e["frames"] for e in index)
    print(f"\n{len(index)} episodes, {frames} frames -> {out}  (skipped {len(skipped)})")


if __name__ == "__main__":
    main()

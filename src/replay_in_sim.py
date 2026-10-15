"""Retarget the extracted hand trajectories onto the Panda in LIBERO and replay them.

Task: LIBERO-Object #4, "pick up the ketchup and place it in the basket".

Retargeting (marker frame on the floor -> LIBERO world frame):
  xy    one similarity transform for the whole dataset (rotation, scale, shift), fitted
        so the average human grasp lands on the ketchup spawn point and the average
        release on the basket centre. Each demo keeps its own variation.
  z     heights are not copied 1:1 (19 cm bottle vs 14.6 cm ketchup, low box vs 14 cm
        basket rim): the shape and timing of the lift are kept but rescaled to clear the
        rim, and the grasp height is mapped into a band where the Panda grips securely.
  rot   the gripper stays pointing down (a parallel gripper can't copy a wrist).
  grip  closes and opens at the moments detected in the video.

From the video: the reach (horizontal path and timing), the whole carry, where the
ketchup starts, the grasp height and the release point. Added by the robot: lining up
above the ketchup and descending onto it, a pause while the gripper closes, staying
above the rim until over the basket, waiting to arrive before opening, lifting away.
Releases that would miss the basket opening are bent inside it over the second half
of the carry (the real box was bigger than the basket); logged as endpoint_fix_m.

The arm is driven with LIBERO's own controller (OSC, delta end-effector actions at
20 Hz), the action space LIBERO policies use, so successful replays become training data.

Usage:
    python src/replay_in_sim.py                        # all episodes, no rendering
    python src/replay_in_sim.py --episodes ep_029 --video
Writes data/sim_replays/: results.csv, ep_XXX.npz, ep_XXX.mp4 (--video), retarget.json
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np


def _default_gl():
    """EGL on Linux, OSMesa (CPU) on WSL2 where EGL usually isn't available.
    On macOS robosuite picks its own renderer, so nothing is set."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        if "microsoft" in Path("/proc/version").read_text().lower():
            return "osmesa"
    except OSError:
        pass
    return "egl"


if _default_gl():
    os.environ.setdefault("MUJOCO_GL", _default_gl())
    if os.environ["MUJOCO_GL"] == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
warnings.filterwarnings("ignore")

SUITE, TASK_ID = "libero_object", 4
KETCHUP_SPAWN = np.array([-0.12, -0.24])  # where the ketchup is in every evaluation layout
BASKET_SPAWN = np.array([0.0, 0.26])      # centre of the basket's spawn region
GRASP_BAND = (0.085, 0.105)               # gripper heights that grip the ketchup securely (m)
BASKET_RIM = 0.141                        # basket rim height (m)
LIFT_CLEARANCE = 0.06                     # ketchup bottom above the rim at the top of the carry
RIM_MARGIN = 0.03                         # stay this far above the rim until over the opening
SAFE_HALF = 0.035                         # release within +-3.5 cm of the basket centre
OPENING_HALF = 0.05                       # "over the opening" = within +-5 cm of the centre
CTRL_HZ = 20
MAX_DELTA = 0.05                          # OSC position output_max (m per step at |a| = 1)
REACH_SECONDS = 3.5                       # how much of the human reach is used
LOOKAHEAD = 3                             # steps the arm aims ahead on the path (controller lag)
GRASP_DWELL = 0.6                         # s, hold still while the gripper closes
RELEASE_DWELL = 0.5
RETREAT_SECONDS = 1.0


# ---------------------------------------------------------------- retargeting

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def carry_ends(traj):
    """Hand xy at lift-off and at release."""
    c = np.where(traj["segment"] == 1)[0]
    return traj["pos"][c[0], :2], traj["pos"][c[-1], :2]


def fit_xy_transform(trajs):
    """Similarity transform: mean hand grasp -> ketchup spawn, mean hand release -> basket.

    Fitted on where the hand opened, not where the bottle came to rest: the bottle
    tipped ~6 cm further after release, which put the sim releases on the basket rim.
    """
    ends = np.array([carry_ends(t) for t in trajs])          # (N, 2, 2)
    g_real, r_real = ends[:, 0].mean(0), ends[:, 1].mean(0)
    v_real, v_sim = r_real - g_real, BASKET_SPAWN - KETCHUP_SPAWN
    scale = np.linalg.norm(v_sim) / np.linalg.norm(v_real)
    ang = np.arctan2(v_sim[1], v_sim[0]) - np.arctan2(v_real[1], v_real[0])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return {"scale": float(scale), "angle_deg": float(np.degrees(ang)), "R": rot.tolist(),
            "real_origin": g_real.tolist(), "sim_origin": KETCHUP_SPAWN.tolist()}


def apply_xy(T, xy):
    xy = np.atleast_2d(xy)
    return (T["scale"] * (np.array(T["R"]) @ (xy - np.array(T["real_origin"])).T)).T + np.array(T["sim_origin"])


def retarget(traj, meta, T, home, basket_xy, speed=1.0):
    """Turn one extracted demo into robot targets at CTRL_HZ.

    Returns (plan, info): plan holds the reach and carry paths (K x 3 arrays), the
    grasp / pre-grasp points and the rim constraint; info holds per-demo numbers.
    speed > 1 plays the human motion faster (same path, less time).
    """
    t, pos, seg = traj["t"], traj["pos"], traj["segment"]
    carry = np.where(seg == 1)[0]
    lift, release = int(carry[0]), int(carry[-1])
    dt = 1.0 / CTRL_HZ

    def resample(t0, t1):
        tt = np.arange(t0, t1, dt * speed)
        return tt, np.stack([np.interp(tt, t, pos[:, k]) for k in range(3)], axis=1)

    ketchup_xy = apply_xy(T, pos[lift, :2])[0]          # where the hand closed on the bottle
    # grasp height as a fraction of the bottle, mapped into GRASP_BAND (found by a sweep:
    # below ~8.5 cm the ketchup drops or hits the basket, near the neck it slips)
    frac = np.clip((pos[lift, 2] / meta["bottle_height_m"] - 0.45) / 0.6, 0.0, 1.0)
    g_sim = float(GRASP_BAND[0] + frac * (GRASP_BAND[1] - GRASP_BAND[0]))
    grasp = np.array([ketchup_xy[0], ketchup_xy[1], g_sim])

    # reach: the hand's horizontal path and timing, warped to start at the robot's home
    # and end above the ketchup (the descent is done robot-side)
    tr, pr = resample(max(0.0, t[lift] - REACH_SECONDS), t[lift])
    xy = apply_xy(T, pr[:, :2])
    w = smoothstep(np.linspace(0, 1, len(tr)))[:, None]
    xy = xy + (1 - w) * (home[:2] - xy[0]) + w * (grasp[:2] - xy[-1])
    pre_z = g_sim + 0.10
    z = home[2] + (pre_z - home[2]) * smoothstep(np.linspace(0, 1, len(tr)))
    reach = np.c_[xy, z]

    # carry: straight from the hand, lift rescaled to clear the rim
    tc, pc = resample(t[lift], t[release] + dt * speed)
    xy = apply_xy(T, pc[:, :2])
    xy = xy + (1 - smoothstep(np.linspace(0, 1, len(tc)) / 0.4))[:, None] * (grasp[:2] - xy[0])
    dz = pc[:, 2] - pc[0, 2]
    peak = max(dz.max(), 0.03)
    z = np.where(dz > 0, g_sim + dz * ((BASKET_RIM + LIFT_CLEARANCE) / peak), g_sim + dz)
    z = np.maximum(z, g_sim + 0.02)                      # never push the ketchup into the table

    # bend the second half of the carry so the release lands inside the basket opening
    rel = xy[-1] - basket_xy
    fix = np.clip(rel, -SAFE_HALF, SAFE_HALF) - rel
    xy = xy + smoothstep((np.linspace(0, 1, len(tc)) - 0.5) / 0.5)[:, None] * fix
    carry_path = np.c_[xy, z]

    info = {"ketchup_xy": ketchup_xy.tolist(), "grasp_z": g_sim, "release_xy": carry_path[-1, :2].tolist(),
            "endpoint_fix_m": float(np.linalg.norm(fix)), "speed": float(speed)}
    plan = {"reach": reach, "grasp": grasp, "pre_grasp": np.r_[grasp[:2], pre_z], "carry": carry_path,
            "basket_xy": basket_xy, "rim_z": g_sim + BASKET_RIM + RIM_MARGIN}
    return plan, info


# ---------------------------------------------------------------- simulation

def make_env(size, cameras=True):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()[SUITE]().get_task(TASK_ID)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    kw = {} if cameras else {"use_camera_obs": False}    # no rendering: ~5x faster
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size, **kw)
    env.seed(0)
    return env, load_init_states(task), task.language


def load_init_states(task):
    """LIBERO's 50 fixed starting layouts for the task (weights_only=False for torch >= 2.6)."""
    import torch
    from libero.libero import get_libero_path

    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(path, weights_only=False)


def reset_with_ketchup(env, init_state, ketchup_xy, settle=10):
    env.reset()
    env.set_init_state(init_state)
    sim = env.env.sim
    q = sim.data.get_joint_qpos("ketchup_1_joint0").copy()
    q[:2] = ketchup_xy
    sim.data.set_joint_qpos("ketchup_1_joint0", q)
    sim.data.set_joint_qvel("ketchup_1_joint0", np.zeros(6))
    sim.forward()
    obs = None
    for _ in range(settle):                              # same 10 no-op steps as lerobot-eval
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    return obs


def rot_action(cur_quat_xyzw, target_R):
    """Axis-angle rotation action that turns the current orientation into target_R."""
    from robosuite.utils import transform_utils as tu

    err = tu.quat2axisangle(tu.mat2quat(target_R @ tu.quat2mat(cur_quat_xyzw).T))
    return np.clip(err / 0.5, -1, 1)


class Runner:
    """Steps the env towards targets and logs actions, states and (optionally) frames."""

    def __init__(self, env, obs, record, hook=None):
        from robosuite.utils import transform_utils as tu

        self.tu, self.env, self.obs, self.record, self.hook = tu, env, obs, record, hook
        self.down_R = tu.quat2mat(obs["robot0_eef_quat"])   # keep the starting, pointing-down orientation
        self.actions, self.states, self.frames, self.phases = [], [], [], []
        self.success_step = None

    def eef(self):
        return self.obs["robot0_eef_pos"]

    def step(self, target, grip, phase):
        o, tu = self.obs, self.tu
        a = np.zeros(7)
        a[:3] = np.clip((target - o["robot0_eef_pos"]) / MAX_DELTA, -1, 1)
        a[3:6] = rot_action(o["robot0_eef_quat"], self.down_R)
        a[6] = grip
        self.states.append(np.r_[o["robot0_eef_pos"], tu.quat2axisangle(o["robot0_eef_quat"]),
                                 o["robot0_gripper_qpos"]])
        if self.record:
            self.frames.append(np.concatenate([o["agentview_image"][::-1], o["robot0_eye_in_hand_image"][::-1]], 1))
        self.actions.append(a)
        self.phases.append(phase)
        if self.hook is not None:            # make_dataset.py records (observation, action) here
            self.hook(o, a, phase)
        self.obs, _, _, _ = self.env.step(a.tolist())
        if self.success_step is None and self.env.check_success():
            self.success_step = len(self.actions) - 1

    def follow(self, path, grip, phase, lead=LOOKAHEAD, rim=None):
        """Track a time-indexed path, aiming a few steps ahead to make up for controller lag.

        rim=(basket_xy, z): until the gripper is over the basket opening, don't go below z,
        so the ketchup can't catch the rim (the descent is delayed, not removed).
        """
        for k in range(len(path)):
            target = path[min(k + lead, len(path) - 1)].copy()
            if rim is not None:
                over = np.all(np.abs(self.eef()[:2] - rim[0]) < OPENING_HALF)
                if k > len(path) // 4 and not over:      # once the ketchup is lifted and moving
                    target[2] = max(target[2], rim[1])
            self.step(target, grip, phase)

    def settle(self, target, grip, phase, tol, max_s, min_s=0.0):
        """Hold a target until the gripper is within tol of it (or max_s runs out)."""
        for k in range(int(max_s * CTRL_HZ)):
            if np.linalg.norm(self.eef() - target) < tol and k >= int(min_s * CTRL_HZ):
                break
            self.step(target, grip, phase)

    def move(self, a, b, seconds, grip, phase):
        n = max(2, int(seconds * CTRL_HZ))
        self.follow(a + (b - a) * smoothstep(np.linspace(0, 1, n))[:, None], grip, phase)


# phase ids: 0 reach (human), 1 align, 2 descend, 3 close, 4 carry (human), 5 arrive, 6 release, 7 retreat

def run_episode(env, init_state, plan, info, record=False, hook=None):
    sp = info.get("speed", 1.0)                 # robot-side moves are sped up by the same factor
    r = Runner(env, reset_with_ketchup(env, init_state, info["ketchup_xy"]), record, hook)
    r.follow(plan["reach"], -1, 0)                                          # human reach
    r.settle(plan["pre_grasp"], -1, 1, tol=0.01, max_s=1.5)                # line up above the ketchup
    r.move(r.eef().copy(), plan["grasp"], 0.8 / sp, -1, 2)                  # straight down
    r.settle(plan["grasp"], -1, 2, tol=0.01, max_s=0.8)
    r.settle(plan["grasp"], 1, 3, tol=0.0, max_s=GRASP_DWELL, min_s=GRASP_DWELL)   # close the gripper
    r.follow(plan["carry"], 1, 4, rim=(plan["basket_xy"], plan["rim_z"]))  # human carry
    r.settle(plan["carry"][-1], 1, 5, tol=0.015, max_s=1.0 / sp)            # arrive before letting go
    r.settle(plan["carry"][-1], -1, 6, tol=0.0, max_s=RELEASE_DWELL, min_s=RELEASE_DWELL)
    r.move(r.eef().copy(), r.eef() + [0, 0, 0.12], RETREAT_SECONDS / sp, -1, 7)  # lift away
    if record:
        o = r.obs
        r.frames.append(np.concatenate([o["agentview_image"][::-1], o["robot0_eye_in_hand_image"][::-1]], 1))
    return {"success": bool(env.check_success()), "success_step": r.success_step,
            "actions": np.array(r.actions), "states": np.array(r.states), "phase": np.array(r.phases),
            "ketchup_end": env.env.sim.data.get_joint_qpos("ketchup_1_joint0")[:3].copy(), "frames": r.frames}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", default="data/trajectories")
    p.add_argument("--out", default="data/sim_replays")
    p.add_argument("--episodes", nargs="+")
    p.add_argument("--video", action="store_true", help="save an mp4 per episode")
    p.add_argument("--init-state", type=int, default=0, help="which of LIBERO's 50 layouts to use")
    p.add_argument("--speed", type=float, default=1.3,
                   help="play the human motion this many times faster (1.3 keeps every episode "
                        "well inside LIBERO's 280-step limit)")
    args = p.parse_args()

    traj_dir, out = Path(args.traj), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(f.stem for f in traj_dir.glob("ep_*.npz"))
    metas = {n: json.loads((traj_dir / f"{n}.json").read_text()) for n in names}
    trajs = {n: dict(np.load(traj_dir / f"{n}.npz")) for n in names}
    T = fit_xy_transform(list(trajs.values()))
    (out / "retarget.json").write_text(json.dumps(T, indent=2))
    print(f"xy transform: scale {T['scale']:.3f}, rotation {T['angle_deg']:.1f} deg, speed x{args.speed}")
    if args.episodes:
        names = [n for n in names if n in args.episodes]

    env, inits, language = make_env(256, cameras=args.video)
    init_state = inits[args.init_state]
    home = reset_with_ketchup(env, init_state, KETCHUP_SPAWN)["robot0_eef_pos"].copy()
    basket_xy = env.env.sim.data.get_joint_qpos("basket_1_joint0")[:2].copy()

    rows = []
    for n in names:
        plan, info = retarget(trajs[n], metas[n], T, home, basket_xy, args.speed)
        res = run_episode(env, init_state, plan, info, record=args.video)
        np.savez_compressed(out / f"{n}.npz", actions=res["actions"], states=res["states"],
                            phase=res["phase"], ketchup_xy=np.array(info["ketchup_xy"]),
                            init_state_index=args.init_state, success=res["success"])
        if args.video and res["frames"]:
            import imageio
            imageio.mimwrite(out / f"{n}.mp4", res["frames"], fps=CTRL_HZ, quality=7, macro_block_size=1)
        row = {"episode": n, "success": int(res["success"]), "steps": len(res["actions"]),
               "success_step": "" if res["success_step"] is None else res["success_step"],
               "ketchup_x": round(info["ketchup_xy"][0], 4), "ketchup_y": round(info["ketchup_xy"][1], 4),
               "release_x": round(info["release_xy"][0], 4), "release_y": round(info["release_xy"][1], 4),
               "grasp_z": round(info["grasp_z"], 4), "endpoint_fix_m": round(info["endpoint_fix_m"], 4),
               "ketchup_end": " ".join(f"{v:.3f}" for v in res["ketchup_end"])}
        rows.append(row)
        print(f"{n}: {'SUCCESS' if res['success'] else 'fail   '}  steps {row['steps']}  "
              f"ketchup ends at {row['ketchup_end']}")

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{sum(r['success'] for r in rows)}/{len(rows)} replays put the ketchup in the basket ({language})")


if __name__ == "__main__":
    main()

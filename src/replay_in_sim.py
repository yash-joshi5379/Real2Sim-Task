"""Retarget the extracted human trajectories onto the Panda in LIBERO and replay them.

Task: LIBERO-Object #4, "pick up the ketchup and place it in the basket".

Retargeting (marker frame on your floor -> LIBERO world frame):

  xy   One similarity transform for the whole dataset (rotation, scale, shift),
       fitted so your average grasp point lands on the ketchup's spawn point and your
       average release point on the basket centre (scale ~1.2: the sim layout is a
       bit bigger than the floor layout). Each demo keeps its own variation: the sim
       ketchup is placed where *that* demo's hand closed on the bottle.
  z    Heights can't be copied 1:1 (19 cm bottle vs 14.6 cm ketchup, a low box vs a
       14 cm basket rim). The *shape and timing* of your lift are kept; its size is
       rescaled so the ketchup clears the rim, and your grasp height (fraction of the
       bottle) is mapped into the band where the Panda grips the ketchup securely.
  rot  The gripper stays pointing down (a parallel-jaw gripper can't copy a wrist).
  grip Closes/opens at the moments detected in the video.

What comes from your video and what the robot adds:
  from you   the reach's horizontal path and timing, the whole carry (path, lift
             profile, speed), where the ketchup starts, grasp height, release point
  robot-side lining up above the ketchup and descending vertically onto it, a pause
             while the gripper closes, staying above the basket rim until over the
             opening, waiting to arrive before opening, and lifting away afterwards
  corrected  releases that would land outside the basket opening are bent inside it
             over the second half of the carry (your box was bigger than the basket);
             the size of each correction is logged as endpoint_fix_m

The arm follows the path with LIBERO's own controller (OSC, delta end-effector
actions at 20 Hz) - the same action space LIBERO policies use, so successful
replays can be turned straight into training data.

Usage (in the conda env with LIBERO):
    python src/replay_in_sim.py                       # all episodes, no rendering (fast)
    python src/replay_in_sim.py --episodes ep_001 ep_029 --video
Outputs in data/sim_replays/: results.csv, ep_XXX.npz (actions, states, phases,
ketchup start), ep_XXX.mp4 with --video, and retarget.json (the fitted transform).
"""

import argparse
import csv
import json
import os
import warnings
from pathlib import Path

import numpy as np

def _default_gl():
    """EGL on a normal Linux box, OSMesa (CPU) on WSL2 where EGL usually isn't available."""
    try:
        if "microsoft" in Path("/proc/version").read_text().lower():
            return "osmesa"
    except OSError:
        pass
    return "egl"


os.environ.setdefault("MUJOCO_GL", _default_gl())
if os.environ["MUJOCO_GL"] == "osmesa":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
warnings.filterwarnings("ignore")

SUITE, TASK_ID = "libero_object", 4
KETCHUP_SPAWN = np.array([-0.12, -0.24])     # centre of target_object_region (BDDL)
BASKET_SPAWN = np.array([0.0, 0.26])         # centre of bin_region (BDDL)
GRASP_BAND = (0.085, 0.105)                  # gripper height range for a secure grasp (m)
BASKET_RIM = 0.141                           # basket rim height (m)
LIFT_CLEARANCE = 0.06                        # ketchup bottom above the rim at the carry's peak
RIM_MARGIN = 0.03                            # ketchup bottom above the rim until it is over the opening
SAFE_HALF = 0.035                            # release must be within +-3.5 cm of the basket centre (x and y)
OPENING_HALF = 0.05                          # "over the opening": gripper within +-5 cm of the centre
CTRL_HZ = 20
MAX_DELTA = 0.05                             # OSC output_max for position (m per step at |a|=1)
REACH_SECONDS = 3.5                          # how much of your reach to use
LOOKAHEAD = 3                                # steps the arm aims ahead on your path (controller lag)
GRASP_DWELL = 0.6                            # s, hold still while the gripper closes
RELEASE_DWELL = 0.5
RETREAT_SECONDS = 1.0


# ----------------------------------------------------------------------------- retargeting

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def carry_ends(traj):
    """Hand position (xy) at lift-off and at release, from the extracted trajectory."""
    c = np.where(traj["segment"] == 1)[0]
    return traj["pos"][c[0], :2], traj["pos"][c[-1], :2]


def fit_xy_transform(trajs):
    """Similarity transform taking your average grasp point to the ketchup's spawn
    point and your average release point to the basket centre.

    The release point is where your hand opened, not where the bottle came to rest
    (it tipped ~6 cm further into the box after you let go).
    """
    ends = np.array([carry_ends(t) for t in trajs])          # (N, 2, 2)
    g_real, r_real = ends[:, 0].mean(0), ends[:, 1].mean(0)
    v_real, v_sim = r_real - g_real, BASKET_SPAWN - KETCHUP_SPAWN
    scale = np.linalg.norm(v_sim) / np.linalg.norm(v_real)
    ang = np.arctan2(v_sim[1], v_sim[0]) - np.arctan2(v_real[1], v_real[0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return {"scale": float(scale), "angle_deg": float(np.degrees(ang)), "R": R.tolist(),
            "real_origin": g_real.tolist(), "sim_origin": KETCHUP_SPAWN.tolist(),
            "fitted_on": "mean hand grasp -> ketchup spawn, mean hand release -> basket centre"}


def apply_xy(T, xy):
    xy = np.atleast_2d(xy)
    out = (T["scale"] * (np.array(T["R"]) @ (xy - np.array(T["real_origin"])).T)).T + np.array(T["sim_origin"])
    return out


def retarget(traj, meta, T, home, basket_xy, speed=1.0):
    """Return (targets (K,3), gripper (K,), phase (K,)) at CTRL_HZ for one demo."""
    t, pos, seg = traj["t"], traj["pos"], traj["segment"]
    carry = np.where(seg == 1)[0]
    L, R = int(carry[0]), int(carry[-1])
    dt = 1.0 / CTRL_HZ

    def resample(t0, t1):
        # speed > 1 plays your motion faster than you did it (same path, less time)
        tt = np.arange(t0, t1, dt * speed)
        return tt, np.stack([np.interp(tt, t, pos[:, k]) for k in range(3)], axis=1)

    ketchup_xy = apply_xy(T, pos[L, :2])[0]        # where your hand closed on the bottle
    h_real = meta["bottle_height_m"]
    g_real = pos[L, 2]
    # Grasp height: where you held the bottle (as a fraction of its height), mapped into
    # the band where the Panda holds the ketchup securely. Tested by sweeping: below
    # ~8.5 cm the ketchup is dropped or knocks the basket, above ~11 cm (the neck) it slips.
    frac = np.clip((g_real / h_real - 0.45) / 0.6, 0.0, 1.0)
    g_sim = float(GRASP_BAND[0] + frac * (GRASP_BAND[1] - GRASP_BAND[0]))
    grasp = np.array([ketchup_xy[0], ketchup_xy[1], g_sim])

    # --- reach: your horizontal path and timing, warped to start at the robot's home
    #     position and end above the ketchup (the descent is added by the runner)
    t0 = max(0.0, t[L] - REACH_SECONDS)
    tr, pr = resample(t0, t[L])
    xy = apply_xy(T, pr[:, :2])
    w = smoothstep(np.linspace(0, 1, len(tr)))[:, None]
    xy = xy + (1 - w) * (home[:2] - xy[0]) + w * (grasp[:2] - xy[-1])
    pre_z = g_sim + 0.10
    z = home[2] + (pre_z - home[2]) * smoothstep(np.linspace(0, 1, len(tr)))
    reach = np.c_[xy, z]

    # --- carry: straight from your hand, with heights rescaled
    tc, pc = resample(t[L], t[R] + dt * speed)
    xy = apply_xy(T, pc[:, :2])
    xy = xy + (1 - smoothstep(np.linspace(0, 1, len(tc)) / 0.4))[:, None] * (grasp[:2] - xy[0])
    dz = pc[:, 2] - pc[0, 2]
    peak = max(dz.max(), 0.03)
    lift_sim = BASKET_RIM + LIFT_CLEARANCE                  # ketchup bottom height at the peak
    z = np.where(dz > 0, g_sim + dz * (lift_sim / peak), g_sim + dz)
    z = np.maximum(z, g_sim + 0.02)                          # don't push the ketchup through the floor

    # Endpoint correction: your box was bigger than the sim basket, so some releases
    # land outside its opening. Bend the second half of the carry so the release ends
    # within SAFE_HALF of the basket centre (most demos need no correction).
    rel = xy[-1] - basket_xy
    fix = np.clip(rel, -SAFE_HALF, SAFE_HALF) - rel
    xy = xy + smoothstep((np.linspace(0, 1, len(tc)) - 0.5) / 0.5)[:, None] * fix
    carry_path = np.c_[xy, z]

    info = {"ketchup_xy": ketchup_xy.tolist(), "grasp_z": g_sim, "carry_peak_real_m": float(peak),
            "release_xy": carry_path[-1, :2].tolist(), "endpoint_fix_m": float(np.linalg.norm(fix)),
            "speed": float(speed),
            "human_steps": int(len(reach) + len(carry_path))}
    return {"reach": reach, "grasp": grasp, "pre_grasp": np.r_[grasp[:2], pre_z],
            "carry": carry_path, "basket_xy": basket_xy,
            "rim_z": g_sim + BASKET_RIM + RIM_MARGIN}, info


# ----------------------------------------------------------------------------- sim

def make_env(size, cameras=True):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    kw = {} if cameras else {"use_camera_obs": False}     # no rendering: ~5x faster, for tuning
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size, **kw)
    env.seed(0)
    return env, load_init_states(task), task.language


def load_init_states(task):
    """LIBERO's fixed initial layouts for a task (works with old and new PyTorch)."""
    import torch
    from libero.libero import get_libero_path

    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(path, weights_only=False)  # LIBERO's own data file (numpy array)


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
    for _ in range(settle):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    return obs


def rot_action(cur_quat_xyzw, target_R):
    """Axis-angle (world frame) that turns the current orientation into target_R."""
    from robosuite.utils import transform_utils as tu

    cur_R = tu.quat2mat(cur_quat_xyzw)
    err = tu.quat2axisangle(tu.mat2quat(target_R @ cur_R.T))
    return np.clip(err / 0.5, -1, 1)


class Runner:
    """Steps the env towards targets, logging actions/states (and frames)."""

    def __init__(self, env, obs, record, hook=None):
        from robosuite.utils import transform_utils as tu

        self.tu, self.env, self.obs, self.record, self.hook = tu, env, obs, record, hook
        self.down_R = tu.quat2mat(obs["robot0_eef_quat"])     # keep the starting, pointing-down orientation
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
        if self.hook is not None:            # e.g. make_dataset.py records (observation, action) here
            self.hook(o, a, phase)
        self.obs, _, _, _ = self.env.step(a.tolist())
        if self.success_step is None and self.env.check_success():
            self.success_step = len(self.actions) - 1

    def follow(self, path, grip, phase, lead=LOOKAHEAD, rim=None):
        """Track a time-indexed path; aim a few steps ahead to make up for controller lag.

        rim=(basket_xy, z): while the gripper is not yet over the basket opening, don't
        go below z, so the ketchup can't catch the rim (your descent is delayed, not removed).
        """
        for k in range(len(path)):
            target = path[min(k + lead, len(path) - 1)].copy()
            if rim is not None:
                over = np.all(np.abs(self.eef()[:2] - rim[0]) < OPENING_HALF)
                started = k > len(path) // 4           # only once the ketchup is lifted and on its way
                if started and not over:
                    target[2] = max(target[2], rim[1])
            self.step(target, grip, phase)

    def settle(self, target, grip, phase, tol, max_s, min_s=0.0, xy_only=False):
        """Hold a target until the gripper is within tol of it (robot-side wait)."""
        for k in range(int(max_s * CTRL_HZ)):
            d = self.eef() - target
            err = np.linalg.norm(d[:2] if xy_only else d)
            if err < tol and k >= int(min_s * CTRL_HZ):
                break
            self.step(target, grip, phase)

    def move(self, a, b, seconds, grip, phase):
        n = max(2, int(seconds * CTRL_HZ))
        path = a + (b - a) * smoothstep(np.linspace(0, 1, n))[:, None]
        self.follow(path, grip, phase)


PHASES = ["reach (human)", "align", "descend", "close", "carry (human)", "settle", "release", "retreat"]


def run_episode(env, init_state, plan, info, record=False, hook=None):
    sp = info.get("speed", 1.0)                 # robot-side moves are sped up by the same factor
    obs = reset_with_ketchup(env, init_state, info["ketchup_xy"])
    r = Runner(env, obs, record, hook)
    r.follow(plan["reach"], -1, 0)                                          # your reach
    r.settle(plan["pre_grasp"], -1, 1, tol=0.01, max_s=1.5)                # line up above the ketchup
    r.move(r.eef().copy(), plan["grasp"], 0.8 / sp, -1, 2)                  # straight down
    r.settle(plan["grasp"], -1, 2, tol=0.01, max_s=0.8)
    r.settle(plan["grasp"], 1, 3, tol=0.0, max_s=GRASP_DWELL, min_s=GRASP_DWELL)   # close the gripper
    r.follow(plan["carry"], 1, 4, rim=(plan["basket_xy"], plan["rim_z"]))  # your carry
    r.settle(plan["carry"][-1], 1, 5, tol=0.015, max_s=1.0 / sp)            # arrive before letting go
    r.settle(plan["carry"][-1], -1, 6, tol=0.0, max_s=RELEASE_DWELL, min_s=RELEASE_DWELL)
    r.move(r.eef().copy(), r.eef() + [0, 0, 0.12], RETREAT_SECONDS / sp, -1, 7)  # lift away
    if record:
        o = r.obs
        r.frames.append(np.concatenate([o["agentview_image"][::-1], o["robot0_eye_in_hand_image"][::-1]], 1))
    ketchup_end = env.env.sim.data.get_joint_qpos("ketchup_1_joint0")[:3].copy()
    return {"success": bool(env.check_success()), "success_step": r.success_step,
            "actions": np.array(r.actions), "states": np.array(r.states), "phase": np.array(r.phases),
            "ketchup_end": ketchup_end, "frames": r.frames}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", default="data/trajectories")
    p.add_argument("--out", default="data/sim_replays")
    p.add_argument("--episodes", nargs="+")
    p.add_argument("--video", action="store_true", help="save an mp4 per episode")
    p.add_argument("--init-state", type=int, default=0, help="which LIBERO initial layout to use")
    p.add_argument("--speed", type=float, default=1.0,
                   help="play your motion this many times faster (1.3 keeps episodes well inside "
                        "LIBERO's 280-step limit)")
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
    obs0 = reset_with_ketchup(env, init_state, KETCHUP_SPAWN)
    home = obs0["robot0_eef_pos"].copy()
    basket_xy = env.env.sim.data.get_joint_qpos("basket_1_joint0")[:2].copy()

    rows = []
    for n in names:
        traj = trajs[n]
        plan, info = retarget(traj, metas[n], T, home, basket_xy, args.speed)
        res = run_episode(env, init_state, plan, info, record=args.video)
        np.savez_compressed(out / f"{n}.npz", actions=res["actions"], states=res["states"],
                            phase=res["phase"], ketchup_xy=np.array(info["ketchup_xy"]),
                            init_state_index=args.init_state, success=res["success"])
        if args.video and res["frames"]:
            import imageio
            imageio.mimwrite(out / f"{n}.mp4", res["frames"], fps=CTRL_HZ, quality=7, macro_block_size=1)
        row = {"episode": n, "success": int(res["success"]), "steps": len(res["actions"]),
               "success_step": res["success_step"] if res["success_step"] is not None else "",
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
    ok = sum(r["success"] for r in rows)
    print(f"\n{ok}/{len(rows)} replays put the ketchup in the basket ({language})")


if __name__ == "__main__":
    main()

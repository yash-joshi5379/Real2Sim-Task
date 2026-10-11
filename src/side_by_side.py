"""Side-by-side video: your phone clip (left) and the Panda replaying it in LIBERO (right).

The two are time-aligned: while the robot follows your reach or carry, the left
side plays the matching moment of your video; during robot-side steps (lining up,
descending, closing the gripper, settling, lifting away) your video holds still on
the matching frame. A caption shows which is which.

Needs the sim replay to have been run with --video (it re-runs the episode here
with rendering if the frames are not available).

Usage:
    python src/side_by_side.py --episodes ep_029 ep_051
Writes data/side_by_side/ep_XXX.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_in_sim as R  # noqa: E402

PHASE_TEXT = ["following your reach", "robot: lining up", "robot: descending", "robot: closing gripper",
              "following your carry", "robot: arriving", "releasing", "robot: lifting away"]


def human_frame_times(phase, t_lift, t_release, reach_start, speed=1.0):
    """For each sim step, the time in your video to show."""
    times, k_reach, k_carry, after = [], 0, 0, 0
    for p in phase:
        if p == 0:
            times.append(reach_start + k_reach * speed / R.CTRL_HZ)
            k_reach += 1
        elif p in (1, 2, 3):
            times.append(t_lift)
        elif p == 4:
            times.append(t_lift + k_carry * speed / R.CTRL_HZ)
            k_carry += 1
        else:
            times.append(t_release + after * speed / R.CTRL_HZ)
            after += 1
    return np.array(times)


def read_frames(path, times, skip_head, height):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = np.clip(np.round(times * fps).astype(int) + skip_head, 0, n - 1)
    cache, out, i, frame = {}, [], -1, None
    for w in sorted(set(wanted.tolist())):
        while i < w:
            ok, f = cap.read()
            if not ok:
                break
            i, frame = i + 1, f
        h, wd = frame.shape[:2]
        w_out = 2 * round(wd * height / h / 2)          # even width for the video encoder
        cache[w] = cv2.resize(frame, (w_out, height), interpolation=cv2.INTER_AREA)
    cap.release()
    for w in wanted:
        out.append(cache[w])
    return out


def label(img, text, sub=None):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.rectangle(img, (0, img.shape[0] - 30), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, sub, (10, img.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--clips", default="data/clips")
    p.add_argument("--traj", default="data/trajectories")
    p.add_argument("--out", default="data/side_by_side")
    p.add_argument("--height", type=int, default=480, help="video height in pixels (even)")
    p.add_argument("--speed", type=float, default=1.0, help="same --speed as used for replay_in_sim.py")
    args = p.parse_args()
    import imageio

    traj_dir, out = Path(args.traj), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(f.stem for f in traj_dir.glob("ep_*.npz"))
    trajs = {n: dict(np.load(traj_dir / f"{n}.npz")) for n in names}
    metas = {n: json.loads((traj_dir / f"{n}.json").read_text()) for n in names}
    T = R.fit_xy_transform(list(trajs.values()))

    env, inits, language = R.make_env(256, cameras=True)
    init_state = inits[0]
    home = R.reset_with_ketchup(env, init_state, R.KETCHUP_SPAWN)["robot0_eef_pos"].copy()
    basket_xy = env.env.sim.data.get_joint_qpos("basket_1_joint0")[:2].copy()

    for n in args.episodes:
        traj, meta = trajs[n], metas[n]
        plan, info = R.retarget(traj, meta, T, home, basket_xy, args.speed)
        res = R.run_episode(env, init_state, plan, info, record=True)
        seg, t = traj["segment"], traj["t"]
        c = np.where(seg == 1)[0]
        t_lift, t_rel = t[c[0]], t[c[-1]]
        times = human_frame_times(res["phase"], t_lift, t_rel, max(0.0, t_lift - R.REACH_SECONDS), args.speed)
        skip = (meta.get("stale_frames_dropped") or [0, 0])[0]
        human = read_frames(Path(args.clips) / f"{n}.mp4", times, skip, args.height)

        frames = []
        for k, ph in enumerate(res["phase"]):
            sim = res["frames"][k][:, :256]                                   # agentview half
            sim = cv2.resize(sim, (args.height, args.height), interpolation=cv2.INTER_CUBIC)
            left = label(human[k], "your demo (phone)")
            right = label(cv2.cvtColor(sim, cv2.COLOR_RGB2BGR), "Panda in LIBERO", PHASE_TEXT[ph])
            frames.append(cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_BGR2RGB))
        verdict = "in the basket" if res["success"] else "not in the basket"
        last = label(cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR), f"{n}: ketchup {verdict}")
        frames += [cv2.cvtColor(last, cv2.COLOR_BGR2RGB)] * R.CTRL_HZ
        imageio.mimwrite(out / f"{n}.mp4", frames, fps=R.CTRL_HZ, quality=7, macro_block_size=1)
        print(f"{n}: {len(frames)} frames -> {out / (n + '.mp4')}  ({verdict})")


if __name__ == "__main__":
    main()

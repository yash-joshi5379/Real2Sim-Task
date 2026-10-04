"""Render LIBERO task scenes so I can pick one to replicate at home.

For every task in the chosen suites this saves:
  - the initial scene from the third-person (agentview) and wrist cameras,
    for a few of the task's fixed initial states
  - one overview sheet per suite (all tasks side by side)
  - tasks.md: each task's instruction, objects, goal and BDDL file

Usage:
    python src/view_scenes.py --list-only                 # just print task names
    python src/view_scenes.py                             # object, spatial, goal suites
    python src/view_scenes.py --suites libero_object --n-init 5
    python src/view_scenes.py --suites libero_goal --tasks 0 3 7
"""

import argparse
import os
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_SUITES = ["libero_object", "libero_spatial", "libero_goal"]


def pick_gl_backend(requested):
    """EGL on a normal Linux box with a GPU, OSMesa (CPU) on WSL2."""
    if requested != "auto":
        return requested
    try:
        if "microsoft" in Path("/proc/version").read_text().lower():
            return "osmesa"
    except OSError:
        pass
    return "egl"


def bddl_block(text, key):
    """Return the non-empty lines inside a top-level (:key ...) block of a BDDL file."""
    start = text.find(f"(:{key}")
    if start == -1:
        return []
    depth = 0
    for end in range(start, len(text)):
        if text[end] == "(":
            depth += 1
        elif text[end] == ")":
            depth -= 1
            if depth == 0:
                break
    body = text[start + len(key) + 2 : end]
    return [line.strip() for line in body.splitlines() if line.strip()]


def summarise_objects(lines):
    """'bowl_1 bowl_2 - akita_black_bowl' lines -> 'akita_black_bowl x2, plate'."""
    counts = Counter()
    for line in lines:
        if " - " not in line:
            continue
        names, obj_type = line.split(" - ", 1)
        counts[obj_type.strip()] += len(names.split())
    return ", ".join(t if n == 1 else f"{t} x{n}" for t, n in counts.items())


def render_task(env, init_states, n_init, settle_steps):
    """Return one [agentview | wrist] image per initial state."""
    frames = []
    for k in range(min(n_init, len(init_states))):
        env.reset()
        obs = env.set_init_state(init_states[k])
        # Let objects settle, as LIBERO's own evaluation does (no motion, gripper open).
        for _ in range(settle_steps):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        agent = obs["agentview_image"][::-1]  # MuJoCo renders upside-down
        wrist = obs["robot0_eye_in_hand_image"][::-1]
        frames.append(np.concatenate([agent, wrist], axis=1))
    return frames


def load_font(size):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def overview_sheet(entries, out_path, cols=2):
    """Tile (caption, image) pairs into one PNG with wrapped captions."""
    from PIL import Image, ImageDraw

    font = load_font(16)
    tiles = []
    for caption, img in entries:
        tile = Image.fromarray(img)
        lines = textwrap.wrap(caption, width=max(20, tile.width // 9))
        canvas = Image.new("RGB", (tile.width, tile.height + 22 * len(lines) + 10), "white")
        canvas.paste(tile, (0, 0))
        draw = ImageDraw.Draw(canvas)
        for j, line in enumerate(lines):
            draw.text((6, tile.height + 5 + 22 * j), line, fill="black", font=font)
        tiles.append(canvas)

    gap = 8
    w = max(t.width for t in tiles) + gap
    h = max(t.height for t in tiles) + gap
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * w, (idx // cols) * h))
    sheet.save(out_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    p.add_argument("--tasks", nargs="+", type=int, help="only these task indices")
    p.add_argument("--n-init", type=int, default=3, help="initial states to render per task")
    p.add_argument("--size", type=int, default=256, help="camera resolution in pixels")
    p.add_argument("--settle-steps", type=int, default=10)
    p.add_argument("--gl", choices=["auto", "egl", "osmesa", "glfw"], default="auto")
    p.add_argument("--out", default="outputs/scenes")
    p.add_argument("--list-only", action="store_true", help="print tasks, render nothing")
    return p.parse_args()


def main():
    args = parse_args()

    # Must be set before robosuite/mujoco are imported.
    os.environ.setdefault("MUJOCO_GL", pick_gl_backend(args.gl))
    if os.environ["MUJOCO_GL"] == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    from PIL import Image
    from libero.libero import benchmark, get_libero_path

    bench = benchmark.get_benchmark_dict()
    out_root = Path(args.out)
    index = ["# LIBERO task index", ""]

    for suite_name in args.suites:
        suite = bench[suite_name]()
        task_ids = [i for i in range(suite.n_tasks) if not args.tasks or i in args.tasks]

        if args.list_only:
            for i in task_ids:
                print(f"{suite_name:15s} {i:2d}  {suite.get_task(i).language}")
            continue

        from libero.libero.envs import OffScreenRenderEnv

        suite_dir = out_root / suite_name
        suite_dir.mkdir(parents=True, exist_ok=True)
        index += [f"## {suite_name}", ""]
        overview = []

        for i in task_ids:
            task = suite.get_task(i)
            bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            bddl_text = Path(bddl_path).read_text()
            print(f"[{suite_name} {i}] {task.language}")

            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path, camera_heights=args.size, camera_widths=args.size
            )
            env.seed(0)
            try:
                frames = render_task(env, suite.get_task_init_states(i), args.n_init, args.settle_steps)
            finally:
                env.close()

            for k, frame in enumerate(frames):
                Image.fromarray(frame).save(suite_dir / f"task{i:02d}_init{k}.png")
            overview.append((f"{i}: {task.language}", frames[0]))

            index += [
                f"### {i}. {task.language}",
                f"- objects: {summarise_objects(bddl_block(bddl_text, 'objects'))}",
                f"- fixtures: {summarise_objects(bddl_block(bddl_text, 'fixtures'))}",
                f"- goal: `{' '.join(bddl_block(bddl_text, 'goal'))}`",
                f"- bddl: `{task.problem_folder}/{task.bddl_file}`",
                f"- images: `{suite_name}/task{i:02d}_init*.png` (left: agentview, right: wrist)",
                "",
            ]

        if overview:
            overview_sheet(overview, out_root / f"{suite_name}_overview.png")

    if not args.list_only:
        (out_root / "tasks.md").write_text("\n".join(index))
        print(f"\nDone. Open {out_root}/ and start with the *_overview.png files.")


if __name__ == "__main__":
    main()

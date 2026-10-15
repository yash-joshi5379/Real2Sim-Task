"""Collect every evaluation in outputs/eval/ into a table and a chart.

Writes outputs/eval/results.md (all runs and checkpoints) and docs/media/results.png
(final checkpoints). Run it after any evaluation:  python src/results.py
"""

import json
from pathlib import Path

# run directory name -> (row label, description, group)
RUNS = {
    "smolvla_phone_blend10":  ("G", "path shapes averaged across demos", "phone"),
    "smolvla_phone_smooth":   ("H", "hand-tracking jitter smoothed out", "phone"),
    "smolvla_phone_eval":     ("D", "each demo's own path", "phone"),
    "smolvla_phone_eval_40k": ("D-40k", "D trained twice as long", "phone"),
    "smolvla_phone_timed":    ("E", "one fixed timing for every demo", "phone"),
    "smolvla_phone_eval4":    ("D4", "each demo replayed in 4 layouts", "phone"),
    "smolvla_phone_spawn4":   ("A2", "bottle at random spawn spots, x4", "phone"),
    "smolvla_phone":          ("A", "bottle where my hand found it", "phone"),
    "act_phone_eval":         ("F", "ACT instead of SmolVLA, D's data", "phone"),
    "smolvla_libero_ref":     ("B", "LIBERO's own robot demos (45)", "reference"),
    "smolvla_scripted_eval":  ("C2", "bottle at the evaluation spot", "control"),
    "smolvla_scripted":       ("C", "bottle where my hand found it", "control"),
    "smolvla_phone_eval_n10": ("D, re-plan every 10 steps", "side test", "side"),
    "smolvla_phone_eval_n5":  ("D, re-plan every 5 steps", "side test", "side"),
}
GROUPS = {"phone": "From my 54 phone videos (no robot demos)", "reference": "Reference",
          "control": "Controls: my pipeline, scripted motion"}


def load(eval_dir):
    rows = {}
    for f in sorted(eval_dir.glob("*/eval_info.json")):
        run, _, ck = f.parent.name.rpartition("_")
        o = json.loads(f.read_text())["overall"]
        lo, hi = o["pc_success_ci95"]
        rows[(run, ck)] = {"n": o["n_success"], "of": o["n_episodes"], "pc": o["pc_success"], "lo": lo, "hi": hi}
    return rows


def table(rows, path):
    order = list(RUNS)
    lines = ["| Run | What changes | Checkpoint | Successes | Success | 95% interval |", "|---|---|---|---|---|---|"]
    for (run, ck), r in sorted(rows.items(), key=lambda kv: (order.index(kv[0][0]) if kv[0][0] in order else 99,
                                                              kv[0][1])):
        label, desc, _ = RUNS.get(run, (run, "", ""))
        lines.append(f"| {label} | {desc} | {ck} | {r['n']}/{r['of']} | {r['pc']:.0f}% | {r['lo']:.0f}-{r['hi']:.0f}% |")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def chart(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = {"phone": "#2a78d6", "reference": "#eb6834", "control": "#1baf7a"}
    ink, ink2, grid, bg = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 6.6), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    y, ticks, heads = 0, [], []
    for g in GROUPS:
        heads.append((y, GROUPS[g]))
        y += 1
        for run, (label, desc, grp) in RUNS.items():
            r = rows.get((run, "last"))
            if grp != g or r is None:
                continue
            ax.barh(y, r["pc"], height=0.62, color=colour[g], zorder=2)
            ax.plot([r["lo"], r["hi"]], [y, y], color=ink2, lw=1.2, zorder=3)
            for x in (r["lo"], r["hi"]):
                ax.plot([x, x], [y - 0.14, y + 0.14], color=ink2, lw=1.2, zorder=3)
            ax.text(r["hi"] + 1.5, y, f"{r['pc']:.0f}%", va="center", color=ink,
                    fontweight="bold" if label in ("G", "B") else "normal")
            ticks.append((y, f"{label}  {desc}"))
            y += 1
        y += 0.4
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], color=ink)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)], color=ink2)
    ax.xaxis.grid(True, color=grid, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(length=0)
    ax.set_xlabel("Success on LIBERO-Object task 4, 50 standard layouts (whiskers: 95% interval)", color=ink2)
    fig.suptitle("Every policy I trained (SmolVLA unless noted), final checkpoints", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=ink)
    fig.tight_layout()
    fig.canvas.draw()
    x0 = min(t.get_window_extent().x0 for t in ax.get_yticklabels()) / fig.bbox.width
    for yy, title in heads:
        fig.text(x0, ax.transData.transform((0, yy))[1] / fig.bbox.height, title, va="center",
                 fontsize=11.5, fontweight="bold", color=ink)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=bg)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    rows = load(root / "outputs/eval")
    table(rows, root / "outputs/eval/results.md")
    chart(rows, root / "docs/media/results.png")

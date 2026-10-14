# Real2Sim: teaching a simulated robot from phone videos of my hand

I recorded myself picking up a bottle and dropping it in a box, 54 times, on a phone.
This repo turns those videos into robot demonstrations for the LIBERO simulator
(LIBERO-Object task 4: *"pick up the ketchup and place it in the basket"*), trains a
vision-language-action model (SmolVLA) on them, and tests it on LIBERO's 50 standard
starting layouts for that task.

| Training data (SmolVLA, 20k steps) | Robot demos used | Success on LIBERO's 50 layouts |
|---|---|---|
| **G: my 54 phone videos, path shapes averaged across demos** | **0** | **96%** (48/50) |
| D: my 54 phone videos, each demo's own path | 0 | 50% (25/50) |
| B: LIBERO's own robot demos of the task (reference) | 45 | 86% (43/50) |
| C2: my pipeline with idealised, scripted motion (control) | 0 | 98% (49/50) |

**In one sentence:** a policy that never saw a robot demonstration learns the task from 54 phone
videos; it succeeds 50% of the time when each demo keeps its own hand path, and **96%** (above
LIBERO's own robot demos) once the demos' path shapes are averaged, because what held it back was
how much a human hand's path varies from one demo to the next.

![one phone demo next to the robot replaying it](docs/media/human_to_robot.gif)

*Left: one of my 54 phone demos. Right: the Panda in LIBERO replaying it after retargeting
(replays like this, recorded with the robot's cameras, are the training data). The caption shows when the robot follows my hand and when it does a robot-side step.*

![policy G completing the task in three different layouts](docs/media/policy_G_success.gif)

*The trained policy (run G) on three of LIBERO's evaluation layouts, real time. It never saw a robot demonstration.*

---

## How it works

```
phone video ──► hand + bottle tracking ──► 3-D gripper path ──► retarget to the Panda
   (54 demos)     (MediaPipe, ArUco)        (metres, on the floor)    (LIBERO coordinates)
                                                                         │
 evaluate on LIBERO's 50 layouts ◄── train SmolVLA ◄── LeRobot dataset ◄─┘ replay in LIBERO,
                                                                          record cameras + actions
```

1. **Recording.** A phone on a fixed stand films a carpet with distractor objects. A protein-shake
   bottle stands in for the ketchup, a shallow box for the basket, and a 100 mm ArUco marker
   (5x5 dictionary, ID 68) printed on A4 paper fixes the scale and the floor plane.
   Four long recordings were cut into 54 clips (`data/clips.csv`).
2. **Extraction** (`src/extract_trajectory.py`).
   - The camera's focal length is estimated from the A4 sheet's edges; the camera pose comes from the marker.
   - The bottle's start and drop points come from the difference between a clip's first and last frames.
   - MediaPipe hand landmarks give the grasp point in every frame. Hands that don't move (my other hand, a foot)
     are dropped by tracking each detection over time.
   - Depth from a single camera is the hard part. While the bottle is carried, the hand's camera ray is
     intersected with the vertical plane through the bottle's start and drop points. During the reach and
     retreat, the marker-based depth is rescaled so it matches at the moments of lift and release.
   - Each clip gives a time-stamped 3-D path, a gripper open/closed signal, and a debug image.
     51/54 were clean; 3 grasped the bottle by the neck and were kept.

   ![extracted gripper path for one demo](docs/media/extraction_ep029.png)

   *One demo after extraction: the tracked grasp point (colour = time, purple → yellow), its shadow on
   the floor, where the bottle started and where it was dropped.*
3. **Retargeting** (`src/replay_in_sim.py`).
   - A similarity transform (scale 1.20, rotation −16.8°) maps the floor onto the table. It is fitted so the
     average human grasp lands on the ketchup and the average release lands in the basket.
   - Heights are rescaled: the real bottle is 19 cm, the sim ketchup 14.6 cm, and the basket rim 14.1 cm.
   - The robot follows my hand's path for the reach and the carry (70% of all control steps).
     Short robot-side steps handle what a phone can't measure precisely: line up above the
     ketchup, descend, close, wait to arrive over the basket, open, lift away.
4. **Dataset** (`src/make_dataset.py`). Every retargeted demo is replayed in LIBERO with both cameras on,
   and each control step is recorded in exactly the format of LeRobot's own LIBERO data (`lerobot/libero`):
   two 256x256 images, the 8-D robot state and the 7-D action. Only replays that succeed are kept (54/54 do).
5. **Training and evaluation.** SmolVLA (`lerobot/smolvla_base`) is fine-tuned with identical settings for
   every run (batch 32, 20k steps, seed 1000) and evaluated with `lerobot-eval` on all 50 layouts.

## Results

All runs: SmolVLA fine-tuned from `lerobot/smolvla_base`, batch 32, seed 1000, evaluated on the 50
standard layouts of LIBERO-Object task 4. The interval is a 95% Wilson interval.

| Run | What the demos are | Episodes | 10k steps | 20k steps (final) |
|---|---|---|---|---|
| B | LIBERO's own robot demos (reference) | 45 | 70% | **86%** (74–93%) |
| A | my motion, bottle where my hand found it | 54 | 22% | 18% (10–31%) |
| A2 | my motion, bottle at random spots in LIBERO's spawn area, x4 | 216 | 22% | 22% (13–35%) |
| C | idealised motion, bottle where my hand found it (control) | 54 | 0% | 0% (0–7%) |
| **D** | **my motion, bottle at the evaluation spot, all 50 layouts** | 54 | 20% | **50%** (37–63%) |
| D4 | as D, each demo replayed in 4 layouts | 216 | 16% | 24% (14–37%) |
| C2 | idealised motion, bottle at the evaluation spot, all 50 layouts (control) | 54 | 88% | **98%** (90–100%) |
| D-40k | as D, trained 40k steps | 54 | 10%* | 48% (35–61%) |
| E | my paths, with one fixed smooth timing for every demo | 54 | 20% | 30% (19–44%) |
| **G** | **average of my 54 path shapes, each demo's own start, end and duration** | 54 | 96% | **96%** (87–99%) |
| F | ACT instead of SmolVLA, on D's data | 54 | 20% | 18% (10–31%) |
| H | as D, with hand-tracking jitter smoothed out (0.75 s filter) | 54 | 24% | 52% (39–65%) |

\* D-40k's learning rate decays more slowly (over 30k steps instead of 20k), so its 10k checkpoint isn't comparable with the others'.

"Idealised motion" (C, C2) replaces my reach and carry with straight, smooth, fixed-duration
moves (reach in 2.7 s; carry = lift, move across, lower) but keeps everything else from each demo:
the grasp height, the release point and the robot-side steps.

![success rate of every policy](docs/media/results.png)

### What the results mean

**1. The pipeline produces data a policy can learn very well.** With idealised motion, the full
pipeline (retargeting, robot-side steps, action labels, dataset format) trains a policy to 98%,
above LIBERO's own human-teleoperated demos (86%). This is close to a best case: those demos are
perfectly consistent. But it rules out a bug in how the data is generated.

**2. Where the bottle is matters more than anything else.** In all 50 evaluation layouts the
ketchup is at exactly the same spot (−0.12, −0.24), and LIBERO's demos all have it there too. My
first datasets put it wherever my hand found the bottle, typically 3 cm and up to 13 cm away.
The policy then had to *find* a narrow bottle by sight from 54 examples; a 2–3 cm error knocks it
over. Moving the bottle to the evaluation spot took idealised motion from 0% to 98% (C → C2) and my
motion from 18% to 50% (A → D). A2 shows that spreading it over LIBERO's spawn area is not enough.

**3. Human motion variation costs about half the successes.** D and C2 differ only in the motion,
and score 50% against 98%. My 54 reaches and carries each take a different path at a different
speed, so the policy learns a blurred average. D's failures look like that: hovering above the
ketchup, slow or hesitant grasps, running out of time. It isn't a lack of training: twice
the training steps (D-40k) gives 48%, the same as D within noise.

Run E tested which part of that variation matters, the path's shape or its timing. It keeps every one of my paths but plays them all at the same smooth speed and duration.
That did **not** help: 30%, against D's 50%. So the cost comes from *where* my hand goes (54
different paths), not *how fast*. If anything, my natural timing helped. A likely reason: people
slow down as they approach an object and before letting go, so the speed fits what the robot sees
at that moment; a uniform schedule throws that away. (The difference between E and D is at the edge of what
50 episodes can show, so I'd call this "no gain", not a proven loss.)

It isn't tracking noise either. The tracked hand jitters by ~6 mm around a smooth curve during
the reach (up to 2 cm), which is a lot next to the robot's ~11 mm per step. Run H filters that
out and keeps my real path shapes: 52%, the same as D. So the variation that hurts is real
variation in how I moved, not measurement error.

**4. Repeating the same motions doesn't help.** D4 replays each of my 54 motions in 4 layouts
that differ by about 1 cm. That adds frames, but no new behaviour, and with the same 20k training
steps each frame is seen 4x less often. It scored 24%, below D. It was still improving (16% →
24%), so it may have been undertrained, although training D for longer didn't help D; with one
seed per run I can't separate the two effects.

**5. Averaging the demos fixes it: 96% from phone video.** Run G keeps everything that is specific
to each demo (where it starts, where the hand grasped and released, how long it took) but gives
every demo the *average* shape of my 54 paths. Each path is split into the straight line from its
start to its end, plus how it bends away from that line; the bends are averaged over normalised time
and put back on each demo's own line. So the motion is still entirely mine (nothing is scripted),
just made consistent. G reaches 96% already at 10k steps, close to the idealised control (98%) and
above LIBERO's own robot demos (86%). This is the most useful result of the project: a small set of
noisy human demos becomes very good training data once their shared shape is extracted and the
demo-to-demo variation is removed. The trade-off: every demo now moves the same way, which suits
this benchmark (the bottle is always in one place) but gives the policy less variety to draw on in
a scene where things move.

![the 54 robot paths in D and in G](docs/media/paths_D_vs_G.png)

*The robot's gripper paths in the training data, all 54 demos. Left: D, each demo follows my own
hand path. Right: G, every demo has the average shape but keeps its own start, end and timing.*

**6. The pretrained VLA matters with this little data.** ACT, a small policy trained from scratch
(apart from an ImageNet image encoder), got 18% on D's data, against SmolVLA's 50%.

### Side test: re-planning more often

SmolVLA predicts 50 actions at a time and normally plays all of them (2.5 s) before looking again.
Re-evaluating D's final model with shorter re-planning intervals:

| Re-plans every | 50 steps (default) | 10 steps | 5 steps |
|---|---|---|---|
| D | 50% | 58% | 42% |

The differences are within the noise of a 50-episode test, so all main results use the default.

### Failure analysis (first 10 evaluation videos, final checkpoints)

![three ways run D fails](docs/media/failure_modes_D.png)

*The three ways run D fails, from its evaluation videos.*

| | Hovers / drifts, never grasps | Knocks the bottle over | Grasps, but drops it beside the basket or runs out of time | Success |
|---|---|---|---|---|
| C | 7 | 2 | 1 | 0 |
| D | 3 | 2 | 2 | 3 |
| E | 4 | 0 | 4 | 2 |

For C and D, the grasp is the hard part: once the ketchup is held, the carry and release usually work.
E is different: it knocked nothing over, but lost more bottles during the carry (dropped beside the basket
or still carrying when time ran out). With 10 videos per run this is a hint, not a measurement.

## What worked and what didn't

Worked:
- **Hand-ray / plane intersection for depth.** Marker-based depth from a single camera was off by up to 40%
  while carrying; intersecting the hand's camera ray with the vertical plane of the carry fixed it.
- **Robot-side grasp and release steps.** Copying the hand's exact grasp failed (see below). Letting the
  human motion drive the large movements while the robot does the last few centimetres made 54/54 replays succeed.
- **Matching the evaluation distribution exactly.** The biggest single gain came from this (finding 2).
- **Control experiments.** C and C2 were what turned "the policy gets ~20%" into a diagnosis.
- **Averaging the path shapes of all demos** (G): 50% → 96%.

Didn't work at first:
- **Heights copied 1:1** failed: the real bottle is taller than the sim ketchup and the box is lower than the basket rim.
- **Fitting the floor → sim transform on where the bottle came to rest** put releases on the basket rim,
  because the bottle tipped ~6 cm further after the hand opened. Fitting on the hand's grasp and release points fixed it.
- **The arm lagged the path by 4–7 cm** and knocked the ketchup over while still moving sideways.
  It now lines up above the ketchup and descends vertically.
- **Grasping where the hand did** (often near the neck) made the ketchup slip. A height sweep gave a secure
  8.5–10.5 cm band, and the human grasp height is mapped into it.
- **More data of the same kind** (A2, D4) and **training longer** (D-40k) did not help.
- **Giving my paths a uniform, smooth timing** (E) did not help either (30% vs 50%).
- **A smaller policy without pretraining** (ACT, F): 18%.
- **Smoothing out hand-tracking jitter** (H): 52%, no change from D.
- **My first explanation for C's 0% was wrong.** I blamed the slow approach above the bottle in the idealised
  motion; C2 has the same approach and scores 98%. C hovered because it never knew exactly where the bottle would be.

## Limitations and future work

- **One task, one seed per run, 50 evaluation episodes.** Differences under ~15 points are within noise.
- **The evaluation always has the ketchup in one place**, so finding 2 is partly about matching the benchmark;
  generalising to new positions would need many more (or more varied) demos.
- **Robot-side grasp and release steps.** A phone can't measure finger contact well enough to drive the gripper
  directly, so ~30% of each episode is scripted. Better hand tracking (e.g. hand-pose models with finger contact)
  could reduce this.
- **Averaging works because the test is narrow.** G's demos all share one path shape. With the bottle in
  different places each time, a middle ground (blending part-way towards the average, or averaging only
  demos that start and end near each other) would keep more variety.
- **Better hand tracking.** 3-D hand-mesh models such as HaMeR or WiLoR would give better depth than a single
  camera with MediaPipe. Run H suggests this wouldn't change the success rate here (removing jitter didn't
  help), but it would matter for tasks that need precise hand positions, and for driving the gripper
  directly instead of with robot-side grasp steps.
- **World models.** The core problem found here is too few, too inconsistent human demos. A world model
  that predicts future video and actions together could generate many more consistent robot demos from
  a handful of human videos (DreamGen / Cosmos-style data generation), or be trained directly on the human
  videos. That needs much more GPU time than this project had, but it attacks exactly the gap measured by D vs C2.

---

## Running it yourself

### Setup

Ubuntu or WSL2 with conda installed:

```bash
bash setup/install_linux.sh             # GPU machine
CPU_ONLY=1 bash setup/install_linux.sh  # laptop / WSL2
conda activate humanoid
```

### Browse LIBERO scenes

```bash
python src/view_scenes.py --list-only   # task names only
python src/view_scenes.py               # render scenes to outputs/scenes/
```

### From phone video to trajectories

1. Put the long recordings in `data/raw/` and list one demo per row in `data/clips.csv`
   (`video,start,end`). Then cut them into `data/clips/ep_001.mp4 ...`:

   ```bash
   python src/cut_clips.py
   ```

2. Extract a gripper trajectory from every clip (downloads MediaPipe's hand model
   to `models/` on first run; about 40 s per clip on a laptop CPU):

   ```bash
   python src/extract_trajectory.py
   ```

   Results go to `data/trajectories/`: one `.npz` + `.json` + `_debug.png` per
   episode, and `summary.csv`. Episodes marked `check` or `failed` in the summary
   need a look at their debug image.

3. Retarget every trajectory onto the Panda and replay it in LIBERO:

   ```bash
   python src/replay_in_sim.py            # all 54, no rendering (~15 s each)
   python src/replay_in_sim.py --episodes ep_029 --video
   ```

   Results go to `data/sim_replays/`: `results.csv` (success per episode),
   `ep_XXX.npz` (actions, states, phases) and `retarget.json` (the fitted floor -> sim transform).

4. Side-by-side video of a demo and its replay (time-aligned):

   ```bash
   python src/side_by_side.py --episodes ep_029 ep_051 --speed 1.3
   ```

5. Build a training dataset (LeRobot format, same layout as `lerobot/libero`):

   ```bash
   python src/make_dataset.py --placement eval --init-state all --out data/lerobot/ketchup_phone_eval  # run D
   python src/make_dataset.py --placement eval --init-state all --dry-run   # check replays only, writes nothing
   ```

   Run G (the best result) is `--placement eval --init-state all --blend 1.0`.
   Options: `--motion human|timed|scripted`, `--placement human|spawn|eval`, `--blend 0..1`
   (pull each path towards the average path shape), `--smooth SECONDS` (remove tracking jitter),
   `--noise STD` (push the `--augment` copies off course and record the corrections),
   `--init-state N|all`, `--augment K` (K extra replays per demo), `--speed` (default 1.3: plays
   the human motion 1.3x faster so every replay finishes well inside LIBERO's 280-step limit).

### Replay results

| | |
|---|---|
| Replays that put the ketchup in the basket | 54 / 54 in layout 0; 54 / 54 with each demo in a different layout (`--init-state all`) |
| Share of each replay driven directly by the human motion (reach + carry) | 70 % of control steps |
| Releases moved to land inside the basket opening | 22 / 54, by 2.1 cm on average (max 5.9 cm) |
| Episode length at 1.3x | success at step 178 (median), 223 (slowest) |

### Training and evaluation

One command per experiment (dataset → training → evaluation of the 10k and final checkpoints):

```bash
bash setup/run_experiment.sh HF_USERNAME phone_blend10 --placement eval --init-state all --blend 1.0  # G
bash setup/run_experiment.sh HF_USERNAME phone_eval --placement eval --init-state all          # D
bash setup/run_experiment.sh HF_USERNAME scripted_eval --motion scripted --placement eval --init-state all  # C2
ONLY=B bash setup/train_both.sh HF_USERNAME                                                    # B (LIBERO's demos)
```

Results are collected in `outputs/eval/results.md` by `setup/eval_all.sh`. Set `WANDB=1` for
live loss curves; text logs go to `logs/`.

### Running on a UCL GPU workstation (no admin rights)

```bash
cd /scratch0/$USER && git clone https://github.com/YOUR_GITHUB/Real2Sim-Task.git   # or: git pull
cd Real2Sim-Task && bash setup/install_ucl.sh          # 15-25 min; reuses saved package versions
source ~/.bashrc && conda activate humanoid
hf auth login && wandb login                           # once per machine
```

End of a session (saves versions, results and logs to git; models and videos to Hugging Face):

```bash
bash setup/backup_session.sh HF_USERNAME
```

## Repository layout

```
src/view_scenes.py         render LIBERO scenes
src/cut_clips.py           long recordings -> one clip per demo
src/extract_trajectory.py  clip -> 3-D gripper path (ArUco + MediaPipe)
src/replay_in_sim.py       retarget a path onto the Panda and replay it in LIBERO
src/side_by_side.py        human clip | robot replay video
src/make_dataset.py        replays -> LeRobot dataset (motion / placement / layout options)
src/reference_episodes.py  finds LIBERO's own demos of the task in lerobot/libero
setup/                     install, train, evaluate and backup scripts
outputs/eval/results.md    all evaluation results
```

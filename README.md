# Humanoid intern challenge

Work in progress.

## Setup

Ubuntu or WSL2 with conda installed:

```bash
bash setup/install_linux.sh             # GPU machine
CPU_ONLY=1 bash setup/install_linux.sh  # laptop / WSL2
conda activate humanoid
```

## Browse LIBERO scenes

```bash
python src/view_scenes.py --list-only   # task names only
python src/view_scenes.py               # render scenes to outputs/scenes/
```

## Data: from phone video to trajectories

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

3. Retarget every trajectory onto the Panda and replay it in LIBERO
   (LIBERO-Object task 4, "pick up the ketchup and place it in the basket"):

   ```bash
   python src/replay_in_sim.py            # all 54, no rendering (~15 s each)
   python src/replay_in_sim.py --episodes ep_029 --video
   ```

   Results go to `data/sim_replays/`: `results.csv` (success per episode),
   `ep_XXX.npz` (the 7-D actions LIBERO policies use, states, phases) and
   `retarget.json` (the fitted floor -> sim transform).

4. Side-by-side video of a demo and its replay (time-aligned):

   ```bash
   python src/side_by_side.py --episodes ep_029 ep_051 --speed 1.3
   ```

5. Build the training dataset (LeRobot format, same layout as `lerobot/libero`):

   ```bash
   python src/make_dataset.py --speed 1.3          # -> data/lerobot/ketchup_from_phone
   ```

   `--speed 1.3` plays the human motion 1.3x faster: LIBERO stops an episode after
   280 steps, and at 1x the slowest replays only succeed at step 278. At 1.3x all 54
   still succeed, at step 178 (median) and 223 (slowest). `--augment K` adds K replays
   per demo with the ketchup moved up to 3 cm.

### Replay results

| | |
|---|---|
| Replays that put the ketchup in the basket | 54 / 54 (LIBERO initial layout 0) |
| Same check on two other LIBERO layouts (17, 33), 10 demos each | 10 / 10, 10 / 10 |
| Share of each replay driven directly by the human motion (reach + carry) | 70 % of control steps |
| Releases moved to land inside the basket opening | 22 / 54, by 2.1 cm on average (max 5.9 cm) |
| Episode length | 222-333 steps at 20 Hz (median 292) |

What did not work at first, and why:

- Copying heights 1:1 failed: the real bottle is 19 cm, the sim ketchup 14.6 cm, and
  the real box is lower than the 14 cm basket rim. The lift is now rescaled to clear the rim.
- Fitting the floor -> sim transform on where the bottle came to rest put releases on the
  basket rim: the bottle tipped ~6 cm further after the hand opened. Fitting on the hand's
  grasp and release points fixed this.
- The arm lagged the path by 4-7 cm, so it reached the ketchup while still moving sideways
  and knocked it over. It now lines up above the ketchup, descends vertically, and waits to
  arrive over the basket before opening.
- Grasping where the hand did (often near the neck) made the ketchup slip; grasping
  below ~8.5 cm made it drop or catch the basket. A height sweep gave a secure 8.5-10.5 cm
  band, and the human grasp height is mapped into it.

## Training and evaluation (SmolVLA, rented GPU)

Two policies, same base model and settings, differing only in where the demos came from:

- **A (this project):** the 54 demos retargeted from phone video.
- **B (reference):** LIBERO's own robot teleoperation demos of the same task, from `lerobot/libero`.

Copy `data/lerobot/ketchup_from_phone/` to the GPU machine, then:

```bash
# A: trained only on demos from phone video
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/ketchup_from_phone \
  --dataset.root=data/lerobot/ketchup_from_phone \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --batch_size=32 --steps=20000 \
  --output_dir=outputs/train/smolvla_phone --job_name=smolvla_phone \
  --policy.device=cuda --policy.push_to_hub=false

# B: trained only on LIBERO's robot demos of the same task
python src/reference_episodes.py        # prints --dataset.episodes=[...]
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/libero \
  --dataset.episodes=[...]   \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --batch_size=32 --steps=20000 \
  --output_dir=outputs/train/smolvla_libero_ref --job_name=smolvla_libero_ref \
  --policy.device=cuda --policy.push_to_hub=false

# Evaluate either one on LIBERO's 50 standard starting layouts for this task
lerobot-eval \
  --policy.path=outputs/train/smolvla_phone/checkpoints/last/pretrained_model \
  --env.type=libero --env.task=libero_object --env.task_ids=[4] \
  --env.observation_height=256 --env.observation_width=256 \
  --eval.n_episodes=50 --eval.batch_size=10 --seed=1000 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```

## Running on a UCL GPU workstation (no admin rights)

Start of a session:

```bash
cd /scratch0/$USER && git clone https://github.com/YOUR_GITHUB/Real2Sim-Task.git   # or: cd Real2Sim-Task && git pull
cd Real2Sim-Task && bash setup/install_ucl.sh          # 15-25 min; reuses saved package versions
source ~/.bashrc && conda activate humanoid
hf auth login && wandb login                           # once per machine
```

One experiment (dataset -> training -> evaluation), inside `tmux`:

```bash
bash setup/run_experiment.sh HF_USERNAME scripted --motion scripted --placement human
```

End of a session (saves versions, results, logs to git; models and videos to Hugging Face):

```bash
bash setup/backup_session.sh HF_USERNAME
```

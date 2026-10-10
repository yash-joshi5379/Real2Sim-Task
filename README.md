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

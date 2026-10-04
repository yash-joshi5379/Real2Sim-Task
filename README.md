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

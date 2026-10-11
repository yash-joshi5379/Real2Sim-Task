"""Exit 0 if the LIBERO ketchup task can be created and rendered (used by install_gpu.sh)."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("MUJOCO_GL", "egl")
import replay_in_sim as R  # noqa: E402

env, inits, _ = R.make_env(64, cameras=True)
obs = R.reset_with_ketchup(env, inits[0], R.KETCHUP_SPAWN, settle=1)
assert obs["agentview_image"].shape == (64, 64, 3) and obs["agentview_image"].max() > 0
print("render ok")

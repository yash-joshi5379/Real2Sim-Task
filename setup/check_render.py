"""Check that the LIBERO ketchup task opens and renders a camera image (exit code 0 if it does)."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import replay_in_sim as R  # noqa: E402  (chooses EGL / OSMesa itself)

env, inits, _ = R.make_env(64, cameras=True)
obs = R.reset_with_ketchup(env, inits[0], R.KETCHUP_SPAWN, settle=1)
assert obs["agentview_image"].shape == (64, 64, 3) and obs["agentview_image"].max() > 0
print(f"render ok (MUJOCO_GL={os.environ.get('MUJOCO_GL')})")

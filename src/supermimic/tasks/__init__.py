"""Register SuperMimic tasks."""

from mjlab.tasks.registry import load_rl_cfg, register_mjlab_task

from .env_cfgs import gencoord_env_cfg, maxcoord_env_cfg
from .evt_env_cfgs import gencoord_evt_env_cfg, maxcoord_evt_env_cfg

# We reuse the RL config from the base tracking task
BASE_TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1"

try:
  register_mjlab_task(
    task_id="SuperMimic-GenCoord-G1",
    env_cfg=gencoord_env_cfg(),
    play_env_cfg=gencoord_env_cfg(play=True),
    rl_cfg=load_rl_cfg(BASE_TASK_ID),
  )
except ValueError:
  # Already registered
  pass

for task_id, factory in (
  ("SuperMimic-GenCoord-EVT", gencoord_evt_env_cfg),
  ("SuperMimic-MaxCoord-EVT", maxcoord_evt_env_cfg),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=factory(),
    play_env_cfg=factory(play=True),
    rl_cfg=load_rl_cfg(BASE_TASK_ID),
  )

try:
  register_mjlab_task(
    task_id="SuperMimic-MaxCoord-G1",
    env_cfg=maxcoord_env_cfg(),
    play_env_cfg=maxcoord_env_cfg(play=True),
    rl_cfg=load_rl_cfg(BASE_TASK_ID),
  )
except ValueError:
  # Already registered
  pass

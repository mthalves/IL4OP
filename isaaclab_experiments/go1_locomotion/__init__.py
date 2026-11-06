"""Unitree Go1 velocity-tracking locomotion, and the ablation of the ContactNet proposals.

The base configuration (``velocity_env_cfg.py``), the MDP terms (``mdp/``) and
the Go1 flat/rough configurations are copied from
``isaaclab_tasks.manager_based.locomotion.velocity`` (IsaacLab 2.3.2) so they
can be modified without touching the vendored IsaacLab sources. Only the robot
asset (``UNITREE_GO1_CFG``) is still imported from ``isaaclab_assets``.

On top of it sit the two proposals that use **ContactNet** (``cnet.py``), a learned
estimator of which feet touch the ground:

* **CEP**, contact probability estimation: the four probabilities are observed by the policy;
* **CCR**, contact consistency reward: the policy is rewarded when the estimate agrees with
  the contacts the simulator reports.

Each can be on or off, which makes four tasks per terrain that share every other setting --
see ``UnitreeGo1RoughEnvCfg`` for how the switches work:

    variant             CEP    CCR    task id
    ------------------  -----  -----  -----------------------------------------------------
    baseline            no     no     IL4OP-Velocity-Rough-Unitree-Go1-v0
    CEP                 yes    no     IL4OP-Velocity-Rough-Unitree-Go1-CEP-v0
    CCR                 no     yes    IL4OP-Velocity-Rough-Unitree-Go1-CCR-v0
    contact framework   yes    yes    IL4OP-Velocity-Rough-Unitree-Go1-ContactFramework-v0

(``Flat`` in place of ``Rough`` for flat terrain) and every one has a ``-Play-v0`` twin.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# The ids are written out in full, one per task: tools/slurm/submit.sh finds the tasks it offers by
# reading the ``*_ENV_ID = "..."`` lines of every package, and a grep for an id has to find it.
FLAT_ENV_ID = "IL4OP-Velocity-Flat-Unitree-Go1-v0"
ROUGH_ENV_ID = "IL4OP-Velocity-Rough-Unitree-Go1-v0"
CEP_FLAT_ENV_ID = "IL4OP-Velocity-Flat-Unitree-Go1-CEP-v0"
CEP_ROUGH_ENV_ID = "IL4OP-Velocity-Rough-Unitree-Go1-CEP-v0"
CCR_FLAT_ENV_ID = "IL4OP-Velocity-Flat-Unitree-Go1-CCR-v0"
CCR_ROUGH_ENV_ID = "IL4OP-Velocity-Rough-Unitree-Go1-CCR-v0"
CONTACT_FRAMEWORK_FLAT_ENV_ID = "IL4OP-Velocity-Flat-Unitree-Go1-ContactFramework-v0"
CONTACT_FRAMEWORK_ROUGH_ENV_ID = "IL4OP-Velocity-Rough-Unitree-Go1-ContactFramework-v0"

#: variant -> (name stem of its classes, {terrain: task id}). The classes are looked up by name,
#: as ``UnitreeGo1{stem}{Rough|Flat}EnvCfg`` and ``UnitreeGo1{stem}{Rough|Flat}PPORunnerCfg``.
VARIANTS = {
    "baseline": ("", {"Flat": FLAT_ENV_ID, "Rough": ROUGH_ENV_ID}),
    "cep": ("CEP", {"Flat": CEP_FLAT_ENV_ID, "Rough": CEP_ROUGH_ENV_ID}),
    "ccr": ("CCR", {"Flat": CCR_FLAT_ENV_ID, "Rough": CCR_ROUGH_ENV_ID}),
    "framework": (
        "ContactFramework",
        {"Flat": CONTACT_FRAMEWORK_FLAT_ENV_ID, "Rough": CONTACT_FRAMEWORK_ROUGH_ENV_ID},
    ),
}

#: variant -> {"Flat": id, "Rough": id}, for whoever wants to loop over the ablation
ENV_IDS = {variant: ids for variant, (_, ids) in VARIANTS.items()}

for variant, (stem, ids) in VARIANTS.items():
    for terrain, module in (("Flat", "flat_env_cfg"), ("Rough", "rough_env_cfg")):
        env_id = ids[terrain]
        env_cls = f"UnitreeGo1{stem}{terrain}EnvCfg"
        runner_cls = f"UnitreeGo1{stem}{terrain}PPORunnerCfg"
        for play in (False, True):
            gym.register(
                id=f"{env_id[:-3]}-Play-v0" if play else env_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{__name__}.{module}:{env_cls}{'_PLAY' if play else ''}",
                    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cls}",
                    "skrl_cfg_entry_point": f"{agents.__name__}:skrl_{terrain.lower()}_ppo_cfg.yaml",
                },
            )

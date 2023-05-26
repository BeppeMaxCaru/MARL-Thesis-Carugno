import marllib as marllib
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

import random
import numpy as np

#SET np.random.seed because the seed cannot be set in rware
# so it just uses current system time!
#This way it works since we control the random generator
np.random.seed(0)
random.seed(0)

#RWARE
env = marl.make_env(
    environment_name="rware",
    map_name="testing_map",
    # force_coop is a boolean parameter that can force the reward the be global
    # If force_coop = True cooperative, if False collaborative scenario
    # Cooperative means same reward for all agents, collaborative individual rewards
    # Originally RWARE is fully cooperative so force_coop = True
    # If cooperative agents are forced to work together, if collaborative they work together
    # but individual rewards instead of global can cause conflict of interests between agents
    force_coop=True,
    #Below are **env_params
    n_agents=4,
    map_size="tiny", 
    difficulty="medium",
)

print(env[0])

obs = env[0].reset() #Used to check that using the seed the obs are always the same
print(obs)

print(env[1])

# initialize algorithm and load hyperparameters
#mappo = marl.algos.mappo(hyperparam_source="test")
#Use IPPO as baseline instead of mappo for the project!
ippo = marl.algos.ippo(hyperparam_source='test')

# can add extra algorithm params. remember to check algo_config hyperparams before use
# mappo = marl.algos.MAPPO(hyperparam_source='common', use_gae=True,  batch_episode=10, kl_coeff=0.2, num_sgd_iter=3)

# build agent model based on env + algorithms + user preference if checked available
#model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

#In RWARE for the fine tuned parameters an FC (fully connected) network is used
#with three layers where the hidden layers contains 128 neurons -> not used if architecture is MLP
#both in the case of with and without parameters sharing
model = marl.build_model(env, ippo, {"core_arch": "mlp", "hidden_state_size": "128", "encode_layer": "256-128"})

print(model[0])
print(model[1])

# start learning + extra experiment settings if needed. remember to check ray.yaml before use
"""
#mappo.fit(env, 
#          model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000}, local_mode=True, num_gpus=1,
#          num_workers=1, share_policy='group', checkpoint_freq=50)
"""

#seed da qualche parte come parametro opzionale
ippo.fit(env, 
          model, #Test
          stop={'timesteps_total': 40000000}, #in RWARE 40 million timesteps used for on-policy
          local_mode=True, 
          num_gpus=1,
          num_workers=4,
          share_policy='group', #Individual so no policy sharing updates for ippo
          checkpoint_freq=5000, #Checkpoint every 1 million steps as in the rware paper
          seed=0,
          #parameters to add to get even more control
          #gamma=0.99, #gamma value is the discount rate in the calculation rewards -> to 0 short-term, to 1 long-term
          #rollout_fragment_lenght=10 #n steps used to calculate discounted reward
)

#ippo.render(env, model, local_mode=True, restore_path='/home/username/ray_results/experiment_name/checkpoint_xxxx')


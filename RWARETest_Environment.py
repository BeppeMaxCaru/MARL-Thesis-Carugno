import marllib as marllib
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

from GraphUsingRay_Environment import RayGraphEnv

# prepare the environment academy_pass_and_shoot_with_keeper
#env = marl.make_env(environment_name="hanabi", map_name="Hanabi-Very-Small")
#env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
"""
#RWARE
env = marl.make_env(environment_name="rware", map_name="testing_map")

# can add extra env params. remember to check env configuration before use
# env = marl.make_env(environment_name='smac', map_name='3m', difficulty="6", reward_scale_rate=15)

# initialize algorithm and load hyperparameters
mappo = marl.algos.itrpo(hyperparam_source="test")

# can add extra algorithm params. remember to check algo_config hyperparams before use
# mappo = marl.algos.MAPPO(hyperparam_source='common', use_gae=True,  batch_episode=10, kl_coeff=0.2, num_sgd_iter=3)

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start learning + extra experiment settings if needed. remember to check ray.yaml before use
mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000}, local_mode=True, num_gpus=1,
          num_workers=1, share_policy='group', checkpoint_freq=50)
"""          

##################################### CUSTOM ENV ############################################

#A nice idea to simplify the environment would be to pass the graph to it externally
#so it doesn't have to be generated inside it
#This way a database of graphs can be built and an environment can be created from one of
#graph of this dataset that can be passed to the environment

#CUSTOM_ENV
ENV_REGISTRY["patrolling"] = RayGraphEnv

env = marl.make_env(environment_name="patrolling", 
                    map_name="patrolling_graph", 
                    num_patrollers=1,
                    num_attackers=0)

# pick mappo algorithms
mappo = marl.algos.mappo(hyperparam_source="test")

# customize model
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})

# start learning
mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000}, local_mode=True, num_gpus=1,
        num_workers=2, share_policy='all', checkpoint_freq=50)

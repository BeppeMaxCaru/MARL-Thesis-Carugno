import marllib as marllib
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

from GraphUsingRay_Environment import RayGraphEnv

##################################### CUSTOM ENV ############################################

#A nice idea to simplify the environment would be to pass the graph to it externally
#so it doesn't have to be generated inside it
#This way a database of graphs can be built and an environment can be created from one of
#graph of this dataset that can be passed to the environment

#CUSTOM_ENV
ENV_REGISTRY["patrolling"] = RayGraphEnv

env = marl.make_env(environment_name="patrolling", 
                    map_name="patrolling_graph",
                    size=10,
                    num_agents=1, 
                    num_patrollers=1,
                    num_attackers=0)

print("ok make env")

# pick mappo algorithms
mappo = marl.algos.ippo(hyperparam_source="test")

print(env)
print(mappo)

#mettendo marl.algos.ippo qualcosa fa senza dare errori!

print("ok mappo init")

# customize model
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})

print("ok model build")

# start learning
#mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000}, local_mode=True, num_gpus=1,
#        num_workers=2, share_policy='group', checkpoint_freq=50, use_opponent_obs=False)

mappo.fit(env, model, stop={'timesteps_total': 100}, local_mode=True, num_gpus=0,
        num_workers=1, share_policy='individual', checkpoint_freq=50)

print("finished training successfully")
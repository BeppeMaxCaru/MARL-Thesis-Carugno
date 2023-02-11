import environment
import graph
import agent
#import gymnasium as gym
import gym
from gym.utils.env_checker import check_env

import stable_baselines3 as sb3
import MARLlib
import ray
import rllib
from ray.rllib.algorithms import ppo

#######################################################
#Test on the basic gym environment -> no multi agent pettingzoo environment
import GraphUsingGym_Environment as gymGraphEnv

env = gymGraphEnv.GymGraphEnv()

print(env.observation_space)

#Flatten observation space
wrapped_env = gym.wrappers.FlattenObservation(env)
check_env(wrapped_env)

#Test with normal environment
env.reset()

for i in range(25):
    obs, reward, done, info = env.step(env.action_space.sample())
    print(reward)
    #Print also agent position
    print(env.current_agent_node)
    
#print(wrapped_env.action_space.sample())
#print(wrapped_env.observation_space.sample())

#Train the agent with tensorboard logging
#sb3.ppo.PPO("MlpPolicy", wrapped_env, verbose=1, tensorboard_log="log_folder_test/").learn(total_timesteps=10000)
#Train the agent without tensorboard logging
#sb3.ppo.PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="log_folder_test_gym/").learn(total_timesteps=100000)


########################################
#Test using ray rllib -> working!
ray.init()

config = ppo.PPOConfig()

print(config.to_dict())

algo = config.build(env=gymGraphEnv.GymGraphEnv)

for i in range(10):
    print(algo.train())


#########################################


#Test with environment with flattened observation space
"""
wrapped_env.reset()
for i in range(250):
    obs, reward, done, info = wrapped_env.step(wrapped_env.action_space.sample())
    print(reward)
    
sb3.ppo.PPO("MlpPolicy", wrapped_env, verbose=1).learn(total_timesteps=10000)
"""

#Test the agent for 10 episodes
"""
model = sb3.ppo.PPO("MlpPolicy", wrapped_env, verbose=1).learn(total_timesteps=10000)

episodes = 10
for i in range(episodes):
    obs = wrapped_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = wrapped_env.step(action)
        #wrapped_env.render()
"""        

##################################################
#Using MARLlib for multi agents

"""
#Train the agent with MARLlib for 1000 iterations
ray.init()
trainer = (
    ray.algorithms.ppo.PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_cpus_for_local_worker=1)
    .env(wrapped_env)
    .build()
)

for i in range(1000):
    trainer.train()

"""

#############################################################
"""
#print("Hello World")
    
#Generate graph
#Uncomment this line to test graph generation
#graphToPatrol = graph.Graph(10, 10)
#Done inside directly into _setup



#Generate environment
env = environment.myEnvironment(num_patrollers=1, num_attackers=0)
#Convert pettingzoo environment to gym environment to make it compatible with rllib
env = gym.make(env)

check_env(env)

#Test the above environment with only one patroller
env.reset()
env.render()
env

    
#Create agent and pass it the action and observation spaces


#Create agent and pass it the action and observation spaces
patroller = agent.PatrollerAgent("Patroller_0", env.action_space, env.observation_space, env.graph, env.target_nodes)

#Pass the agent to the environment
patroller = env.add_agent(patroller)

#Train the agent
patroller._train(env)

"""

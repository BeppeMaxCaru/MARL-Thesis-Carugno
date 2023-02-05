import environment
import graph
import agent
#import gymnasium as gym
import gym
from gym.utils.env_checker import check_env

import stable_baselines3 as sb3

#######################################################
#Test on a basic gym environment

import GraphUsingGym_Environment as gymGraphEnv

env = gymGraphEnv.GymGraphEnv()
#Flatten observation space
wrapped_env = gym.wrappers.FlattenObservation(env)
check_env(wrapped_env)

#Test the above environment
wrapped_env.reset()

for i in range(25):
    obs, reward, done, info = wrapped_env.step(wrapped_env.action_space.sample())
    print(reward)
    
#print(wrapped_env.action_space.sample())
#print(wrapped_env.observation_space.sample())

#Train the agent with tensorboard logging
#sb3.ppo.PPO("MlpPolicy", wrapped_env, verbose=1, tensorboard_log="log_folder_test/").learn(total_timesteps=10000)
#Train the agent without tensorboard logging
sb3.ppo.PPO("MlpPolicy", wrapped_env, verbose=1).learn(total_timesteps=10000)





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

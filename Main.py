import environment
import graph
import agent
#import gymnasium as gym
import gym
from gym.utils.env_checker import check_env

#######################################################
#Test on a basic gym environment

import GraphUsingGym_Environment as gymGraphEnv

env = gymGraphEnv.GymGraphEnv()
#Flatten observation space
wrapped_env = gym.wrappers.FlattenObservation(env)

check_env(wrapped_env)

print("Hello World")






#############################################################

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

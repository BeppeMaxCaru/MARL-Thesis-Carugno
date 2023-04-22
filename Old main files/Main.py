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
#wrapped_env = gym.wrappers.FlattenObservation(env)
wrapped_env = env
check_env(wrapped_env)

#Test with normal environment
env.reset()

for i in range(25):
    #Save previous node to print transition
    previous_agent_node = str(env.current_agent_node)
    #Action
    obs, reward, done, info = env.step(env.action_space.sample())
    #Print also action result
    current_agent_node = str(env.current_agent_node)
    agent_reward = str(reward)
    print("Movement from node: " + previous_agent_node + 
          " to node: " + current_agent_node +
          " with a reward of: " + agent_reward)

env.graph.draw()

print("Setup and checks completed succesfully")
print("######################################")

#print(wrapped_env.action_space.sample())
#print(wrapped_env.observation_space.sample())

#Train the agent with tensorboard logging
#sb3.ppo.PPO("MlpPolicy", wrapped_env, verbose=1, tensorboard_log="log_folder_test/").learn(total_timesteps=10000)
#Train the agent without tensorboard logging
#sb3.ppo.PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="log_folder_test_gym/").learn(total_timesteps=100000)


"""

########################################
#Test using ray rllib -> working!
ray.init()

config = ppo.PPOConfig()

print(config.to_dict())

algo = config.build(env=gymGraphEnv.GymGraphEnv)

for i in range(10):
    print(algo.train())


"""
#########################################
print("Graph check")
wrapped_env.graph.draw()

#Pick algo and model
print("Model selection")
model = sb3.ppo.PPO("MultiInputPolicy", wrapped_env, verbose=1)
#Train model
print("Training the model")
model.learn(total_timesteps=1000)

#Evaluate policy
print("Evaluating policy")
#NB The policy is stored into the model and it is accessible through:
#policy = model.policy -> it is the value of the weights of the underlying neural network
#and it can be exported and manipulated using either TensorFlow or PyTorch in Stable Baselines
#however the suggested way is using PyTorch according to the documentation
mean_reward, std_reward = sb3.common.evaluation.evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#Alternative method to evaluate the policy:
#policy = model.policy
#mean_reward, std_reward = sb3.common.evaluation.evaluate_policy(policy, model.get_env(), n_eval_episodes=10)
print("Mean reward: " + str(mean_reward))
print("Std reward: " + str(std_reward))

print(model.policy)

print("Testing the model")
#Test model and save policy
model_env = model.get_env()
obs = model_env.reset()
episodes = 50
for i in range(episodes):
    #obs = wrapped_env.reset()
    #done = False
    
    previous_agent_node = str(obs['current_agent_node'])[1:6]
    
    #while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = model_env.step(action)
    
    current_agent_node = str(obs['current_agent_node'])[1:6]
    
    print("Agent moves from: " + previous_agent_node +
          " to: " + current_agent_node +
          " with reward: " + str(rewards))
    #wrapped_env.render()
    
    #if done:
    #    obs = model_env.reset()

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

import tensorflow as tf
import tf_agents
import stable_baselines3
import MARLlib
import numpy as np

#Create the same patroller agent using the tf_agents package
class PatrollerAgent(tf_agents.agents.TFAgent):
    def __init__(self, name, action_space, observation_space, graph, target_nodes):
        self.graph = graph
        self.target_nodes = target_nodes
        
        #FIX HERE THE BELOW INSTRUCTIONS
        ######################################################
        #Given pettingzoo action_space and observation_space, create tf_agents time_step_spec and action_spec        
        #Make action_space compatible with tf_agents and MARLlib
        actions = tf_agents.specs.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=4)        
        #Make observation_space from environment class compatible with tf_agents and MarLlib
        time_step_spec = tf_agents.trajectories.TimeStep(step_type=tf_agents.trajectories.StepType.FIRST, reward=0, discount=1, observation=observation_space)
        #Call the parent class's constructor
        super(PatrollerAgent).__init__(name, time_step_spec=time_step_spec, action_spec=actions, policy=None, collect_policy=None, train_sequence_length=None)
        
        ######################################################
        
        #Example of to set up the agent's RL algorithm
        #Use PPO as reinforcement learning algorithm from MARLlib
        self.rl_algorithm = MARLlib.PPO(actions, observations)
    #Step function using tf-agents for the agent using the RL algorithm selected to choose an action
    def step(self, observation):
        #Use the RL algorithm to choose an action
        action = self.rl_algorithm.choose_action(observation)
        #Return an action
        return {"action" : action}
    #Train function using tf-agents for the agent using the RL algorithm selected to train the agent
    def _train(self, environment):
        #Train the RL algorithm
        self.rl_algorithm.train(environment)

"""
#Here using stable_baselines3
#Create the same patroller agent using the stable_baselines3 package
class PatrollerAgent(stable_baselines3.common.base_class.BaseAlgorithm):
    def __init__(self, name, action_space, observation_space, graph, target_nodes):
        self.graph = graph
        self.target_nodes = target_nodes
        
        super().__init__(name, action_space, observation_space)
        #Example of to set up the agent's RL algorithm
        #Use PPO as reinforcement learning algorithm
        self.rl_algorithm = MARLlib.PPO(action_space, observation_space)
    #step function using stable_baselines3 for the agent using the RL algorithm selected to choose an action
    def step(self, observation):
        #Use the RL algorithm to choose an action
        action = self.rl_algorithm.choose_action(observation)
        #Return an action
        return {"action" : action}
    #Train function using tf-agents for the agent using the RL algorithm selected to train the agent
    def _train(self, environment):
        #Train the RL algorithm
        self.rl_algorithm.train(environment)
"""

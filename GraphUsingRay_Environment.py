import ray
import gymnasium as gym
import numpy as np
import random

from ray.rllib.env import MultiAgentEnv

import graph

class RayGraphEnv(MultiAgentEnv):
    
    def __init__(self, size=10, num_patrollers=1, num_attackers=0):
        
        #Define size of the squared grid
        self._graph_size = size
        self._target_nodes_locations = {}
        
        # Define the possible agents in the environment
        self.num_patrollers = num_patrollers
        self.num_attackers = num_attackers
        
        #Agents ids for Ray
        self.agents = set()
        #Add patrollers ids
        for i in range(num_patrollers):
            self.agents.add("Patroller_" + str(i) + "_ID")
        #Add attackers ids
        for i in range(num_attackers):
            self.agents.add("Attacker_" + str(i) + "_ID")
        
        self._agent_ids = set(range(self.agents))
        self._observations = {}
        self._infos = {}
        
        #Dictionary mapping agents to their observations
        self.observation_space = {
            agent: gym.spaces.Dict() for agent in self.agents
        }
        
        #Assign to each agent's observation space its location
        for agent in self.agents:
            self.observation_space[agent]["current_agent_node_location"] = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self._graph_size, self._graph_size]), dtype=np.int32)
        
        #Assign to each agent the possibility to move: Stay, Up, Down, Left, Right
        self.action_space = {
            agent: gym.spaces.Discrete(5) for agent in self.agents
        }
        
    def reset(self):
                
        #Reset observation space for each agent
        self.terminated = set()
        self.truncated = set()
        self._observations = {}
        self._infos = {}
        
        #Generate new graph
        self._generate_new_graph()
        
        for agent in self.agents:
            obs[agent] = {"current_agent_node_location": np.array([0, 0], dtype=np.int32)}
        return obs
        
    def step(self):
        ...
        
    def _generate_new_graph(self):
        
        #Generate new graph
        self._graph = graph.Graph(self._graph_size, self._graph_size)
        
        #Assign random starting location to each agent
        for agent in self.agents:
            self.observation_space[agent]["current_agent_node_location"] = random.choice(list(self._graph.G.nodes))
        
        #Retrieve new list of target nodes (nodes with red color)
        self.target_nodes = [node for node in self.graph.G.nodes if self.graph.G.nodes[node]['color'] == 'red']
        #Determine the position of each target node and assign it to a dictionary for gym observation space
        self.target_nodes_positions_dict = {node: self.graph.G.nodes[node]['position'] for node in self.target_nodes}
        #Create dictionary to convert node to index
        self.node_to_index = {node: index for index, node in enumerate(self.graph.G.nodes)}
        
        #Pass to each agent the targets locations
        for agent in self.agents:
            self.observation_space[agent]['target_node_' + str(i) + '_position'] = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.graph_size, self.graph_size]), dtype=np.int32)
        
        
#Testing
env = RayGraphEnv()

print(env.patrollers)
        
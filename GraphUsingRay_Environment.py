import ray
import gymnasium as gym
import numpy as np
import random

from ray.rllib.env import MultiAgentEnv

from ray.rllib.algorithms.ppo import PPOConfig

from ray import tune
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

import graph

class RayGraphEnv(MultiAgentEnv):
    
    def __init__(self, env_config=None, size=10, num_patrollers=1, num_attackers=0):
        
        test_params_from_config = env_config.get("test", gym.spaces.Box(-1.0, 1.0, shape=(1, )))
        test_obs = env_config.get("observation_space", gym.spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.int32))
        
        #Define size of the squared grid
        self._graph_size = size
                
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
        
        self._agent_ids = set(range(len(self.agents)))
        self._observations = {}
        self._infos = {}
        
        #Assign to each agent's observation space its location
        #for agent in self.agents:
        #    self.agents_to_obs_mapping[agent]["agent_starting_node_location"] = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self._graph_size, self._graph_size]), dtype=np.int32)
        
        self.observation_space = gym.spaces.Dict()
        
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
        
        #Reset target nodes locations dictionary
        self._target_nodes_locations = {}
        #Collect target nodes locations
        self._get_target_nodes_locations()
        
        #Reset agents starting node positions dictionary
        self._agents_starting_nodes_locations = {}
        #Assign random starting location to agents
        self._set_agents_random_starting_locations()
        
        #Reset observations for all agents once reset method is called
        #This is the equivalent of observation space
        self._agents_to_obs_mapping = {}
        self.observation_space = {}
        
        #Set observations spaces for all agents using mapping
        self._set_obs_spaces_for_all_agents()
        
        return self.observation_space
        
    def step(self, action_dict):
        
        obs_dict = {}
        reward_dict = {}
        done_dict = {"__all__": False}
        info_dict = {}
        
        for agent_id, action in action_dict.items():
            
            Agent_ID = agent_id
            # Get the current agent's node location
            agent_starting_node = tuple(self._agents_to_obs_mapping[Agent_ID][Agent_ID + ": current position"])
            new_agent_location = agent_starting_node
            
            #action = actions[0]
            if action == 0:
                #Stay in the same node
                pass
            elif action == 1:
                #Move to the node above in the graph if edge going to it exists
                #Check if the node above exists and if there is an edge going to it
                if (agent_starting_node[0] > 0) and (self._graph.G.has_edge(agent_starting_node, (agent_starting_node[0]-1, agent_starting_node[1]))):
                    #Move to the node above
                    new_agent_location = (agent_starting_node[0]-1, agent_starting_node[1])
            elif action == 2:
                #Move to the node down in the graph if edge exists
                #Check if the node below exists and if there is an edge going to it
                if (agent_starting_node[0] < self._graph.n-1) and (self._graph.G.has_edge(agent_starting_node, (agent_starting_node[0]+1, agent_starting_node[1]))):
                    #Move to the node below
                    new_agent_location = (agent_starting_node[0]+1, agent_starting_node[1])
            elif action == 3:
                #Move to the node on the left in the graph if edge exists
                #Check if the node to the left exists and if there is an edge going to it
                if (agent_starting_node[1] > 0) and (self._graph.G.has_edge(agent_starting_node, (agent_starting_node[0], agent_starting_node[1]-1))):
                    #Move to the node to the left
                    new_agent_location = (agent_starting_node[0], agent_starting_node[1]-1)
            elif action == 4:
                #Move to the node on the right in the graph if edge exists
                #Check if the node to the right exists and if there is an edge going to it
                if (agent_starting_node[1] < self._graph.m-1) and (self._graph.G.has_edge(agent_starting_node, (agent_starting_node[0], agent_starting_node[1]+1))):
                    #Move to the node to the right
                    new_agent_location = (agent_starting_node[0], agent_starting_node[1]+1)
            else:
                #Invalid action
                raise ValueError("Invalid action: {}".format(action))
            
            #Update current node of the agent
            #New agent position = node
            
            # Check if the current agent is on a target node
            pos = tuple(self._agents_to_obs_mapping[Agent_ID][Agent_ID + ": current position"])
            if pos in self._target_nodes_locations:
                # Give reward of 1 if agent is on a target node
                reward_dict[Agent_ID] = 1
                done_dict[Agent_ID] = False
            else:
                reward_dict[Agent_ID] = 0
                done_dict[Agent_ID] = False
                
            # Update the observation dictionary for this agent -> update only his position
            #since targets are static
            obs_dict[Agent_ID] = {
                Agent_ID + ": current position": np.array([new_agent_location[0], new_agent_location[1]], dtype=np.int32),
            }
            i = 0
            for target in self._target_nodes_locations:
                obs_dict[Agent_ID][Agent_ID + ": target_node_" + str(i) + "_location"] = np.array([target[0], target[1]], dtype=np.int32)
                i = i + 1
        
        return obs_dict, reward_dict, done_dict, info_dict
        
    def _generate_new_graph(self):
        
        #Generate new graph
        self._graph = graph.Graph(self._graph_size, self._graph_size)
    
    def _get_target_nodes_locations(self):
        #Collect target nodes locations
        for node in self._graph.G.nodes:
            if self._graph.G.nodes[node]['color'] == 'red':
                #self._target_nodes_locations[node] = node
                #print(node)
                self._target_nodes_locations[node] = node
    
    def _set_agents_random_starting_locations(self):
        for agent_ID in self.agents:
            random_node = random.choice(list(self._graph.G.nodes))
            self._agents_starting_nodes_locations[agent_ID] = random_node
            
    def _set_obs_spaces_for_all_agents(self):
        #Create the observation dictionary for each agent
        #and then fill it by adding the agent position and the locations of the target nodes
        for agent_ID in self.agents:
            #Assign observation dictionary to each agent
            self._agents_to_obs_mapping[agent_ID] = {}
            #Add agent starting random location            
            starting_node = self._agents_starting_nodes_locations[agent_ID]
            self._agents_to_obs_mapping[agent_ID][agent_ID + ": current position"] = np.array([starting_node[0], starting_node[1]], dtype=np.int32)
            #Add target nodes locations to each agent obs space
            i = 0
            for target in self._target_nodes_locations:
                self._agents_to_obs_mapping[agent_ID][agent_ID + ": target_node_" + str(i) + "_location"] = np.array([target[0], target[1]], dtype=np.int32)
                i = i + 1
        
        self.observation_space = dict(self._agents_to_obs_mapping)
    
#Testing without Ray
#to just check that the environment is compiling and is correct
"""
env = RayGraphEnv()

print(env.num_patrollers)
print(env.agents)
obs = env.reset()
print(env._agents_starting_nodes_locations)
print("mapping")
print(env._agents_to_obs_mapping)

for i in range(3):
    obs = env.reset()
    print("New iteration\n")
    print(obs['Patroller_0_ID'])
    print(obs['Patroller_0_ID']['Patroller_0_ID: current position'])
    print(obs['Patroller_0_ID']['Patroller_0_ID: target_node_0_location'])
    print(obs['Patroller_0_ID']['Patroller_0_ID: target_node_1_location'])
    print(obs['Patroller_0_ID']['Patroller_0_ID: target_node_2_location'])
    print(obs['Patroller_0_ID']['Patroller_0_ID: target_node_3_location'])
    print(obs['Patroller_0_ID']['Patroller_0_ID: target_node_4_location'])
    print("\nstart step\n")
    obs_dict, rew, done, info = env.step(action_dict={'Patroller_0_ID': 1})
    print(obs_dict)
    print(rew)
    print(done)
    print(info)
    print("stop step\n")
"""

#Testing with Ray

def env_creator(env_config):
    return RayGraphEnv(env_config)

ray.init()

register_env("RayEnv", env_creator)

config = (
    PPOConfig().
    environment(
        env="RayEnv"
        """
        ,
        env_config={
            "test": gym.spaces.Box(-5.0, 5.0, (1, )),
            "observation_space": gym.spaces.Box(low=np.array([0, 0]), high=np.array([100, 100]), dtype=np.int32)
        }
        """
    )
    .rollouts(num_rollout_workers=1)
)

config = (
    PPOConfig().
    environment(
        env="RayEnv",
    )
    .rollouts(num_rollout_workers=1)
)

algo = config.build()

for i in range(5):
    results = algo.train()
    print(results)
    
ray.shutdown()

        
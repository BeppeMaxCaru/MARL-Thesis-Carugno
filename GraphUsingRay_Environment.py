import ray
import gym as gym
import numpy as np
import random

from ray.rllib.env import MultiAgentEnv

import graph

policy_mapping_dict = {
    "patrolling_graph": {
        "description": "patrolling_test",
        "team_prefix": ("Patroller_", "Attacker_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

REGISTRY = {}
#REGISTRY["patrolling_graph"]

class RayGraphEnv(MultiAgentEnv):
    
    #env_config is a configuration dictionary containing the parameters to use to setup the env
    def __init__(self, env_config):
        
        #Configuration file settings + check
        print("env config parameters:")
        #Define size of the squared grid        
        self._graph_size = env_config["size"]
        print(self._graph_size)
        # Define the possible agents in the environment
        self.num_patrollers = env_config["num_patrollers"]
        print(self.num_patrollers)
        self.num_attackers = env_config["num_attackers"]
        print(self.num_attackers)
                        
        #Agents
        self.agents = []
        #Add patrollers ids
        for i in range(self.num_patrollers):
            self.agents.append("Patroller_" + str(i) + "_ID")
        #Add attackers ids
        for i in range(self.num_attackers):
            self.agents.append("Attacker_" + str(i) + "_ID")
        print(self.agents)
        
        self.num_agents = len(self.agents)
        print(len(self.agents))
        
        #Creating graph directly here instead of each time in reset!
        #This is the correct way to do so!
        #See RWARE example in MARLlin where Warehouse is initialized in __init__ and 
        #not in reset!!!!!!
        
        #Generate new graph        
        self._graph = self._generate_new_graph(self._graph_size)
        #Collect new target nodes locations from graph as a dictionary
        self._target_nodes_locations = self._get_target_nodes_locations(self._graph)
        
        self.observation_space = gym.spaces.Dict({
            #"obs" is the agent location -> do refactoring! 
            "obs": gym.spaces.Box(low=np.array([0, 0]), high=np.array([self._graph_size, self._graph_size]), dtype=np.int32),
        })
        #Add to the observation space a gym space for each target node
        for i, target_node in enumerate(self._target_nodes_locations):
            self.observation_space["target_node_" + str(i) + "_location"] = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self._graph_size, self._graph_size]), dtype=np.int32)
          
        #Same action space for all agents so I just need to declare it once (for now)
        #5 possible actions: Stay, Up, Down, Left, Right
        self.action_space = gym.spaces.Discrete(5)
        #Assign to each agent the possibility to move: Stay, Up, Down, Left, Right
        """
        self.action_space = {
            agent: gym.spaces.Discrete(5) for agent in self.agents
        }
        """
        print("finished init successfully")
                
    def reset(self):
        
        obs = {}
        
        #Assign new random starting location to agents as a dictionary
        self._agents_starting_nodes_locations = self._set_agents_random_starting_locations(self._graph)
        print(self._agents_starting_nodes_locations)
        
        #Reset observations for all agents once reset method is called
        #This is the equivalent of observation space
        self._agents_to_obs_mapping = {}
        
        #Set observations spaces for all agents using mapping
        self._set_obs_spaces_for_all_agents()
        print(self._agents_to_obs_mapping)

        ########### NEW ############
        for i, agent in enumerate(self.agents):
            #Assign to agent starting node
            obs[agent] = {
                "obs": self._agents_starting_nodes_locations[agent]
            }
            #Assign to agent target nodes locations
            for j, target in enumerate(self._target_nodes_locations):
                obs[agent]["target_node_" + str(j) + "_location"] = self._target_nodes_locations[target]
            print(obs)
        
        return obs
                
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
    
    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 1000,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
    
    ############## PRIVATE FUNCTIONS #############
        
    def _generate_new_graph(self, side_dim_of_squared_graph):
        #Generate new graph
        return graph.Graph(side_dim_of_squared_graph, side_dim_of_squared_graph)
    
    def _get_target_nodes_locations(self, xnetwork_graph):
        dict_to_return_with_target_nodes_locations = {}
        #Collect target nodes locations
        for node in xnetwork_graph.G.nodes:
            if xnetwork_graph.G.nodes[node]['color'] == 'red':
                #print(node)
                dict_to_return_with_target_nodes_locations[node] = node 
        return dict_to_return_with_target_nodes_locations
    
    def _set_agents_random_starting_locations(self, networkx_graph):
        random_starting_nodes_locations = {}
        for agent_ID in self.agents:
            random_starting_nodes_locations[agent_ID] = random.choice(list(networkx_graph.G.nodes))
        return random_starting_nodes_locations
            
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
        
    def _set_new_correct_obs(self):
        obs = {}
        for i, agent_ID in enumerate(self.agents):
            current_agent_obs_space = {}
            #Add starting location
            obs[agent_ID] = {"obs": np.array([starting_node[0], starting_node[1]], dtype=np.int32)}
            #Add target nodes location            


#Testing

fake_env_config = {
    "size": 10,
    "num_patrollers": 1,
    "num_attackers": 0
}

grafo = RayGraphEnv(fake_env_config)
grafo.reset()
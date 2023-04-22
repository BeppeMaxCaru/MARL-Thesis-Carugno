import ray
import gym as gym
import numpy as np
import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import graph

policy_mapping_dict = {
    "patrolling_graph": {
        "description": "patrolling_test",
        "team_prefix": ("Patroller_", "Attacker_"),
        "all_agents_one_policy": False, #Defines if agent have a shared policy or each one has its own
        "one_agent_one_policy": True, #Defines if each agent should have its own policy or not
    },
}

REGISTRY = {}
#REGISTRY["patrolling_graph"]



class RayGraphEnv(MultiAgentEnv):
    
    #env_config is a configuration dictionary containing the parameters to use to setup the env
    def __init__(self, env_config):
        
        #Configuration file settings + check
        #Define size of the squared grid        
        self._graph_size = env_config["size"]
        # Define the possible agents in the environment
        self.num_patrollers = env_config["num_patrollers"]
        self.num_attackers = env_config["num_attackers"]
        
        print(env_config)        
        
        #Add patrollers
        self.agents = ["Patroller_" + str(i) for i in range(self.num_patrollers)]
        #Add attackers
        self.agents = self.agents + ["Attacker_" + str(i) for i in range(self.num_attackers)]
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
        
        #Since I cannot use a nested dict in obs so I flatten it to a big Box
        #Max values are minimum and maximum graph size with shape as explained below
        self.observation_space = gym.spaces.Dict({
            #shape = (agents positions and targets  positions) * 2 since both are tuples of 2 elems
            "obs": gym.spaces.Box(low=0, high=self._graph_size, shape=(2 + len(self._target_nodes_locations)*2,), dtype=np.int32),
            #"opponent_obs": None,
        })
        
        #Same action space for all agents so I just need to declare it once (for now)
        #5 possible actions: Stay, Up, Down, Left, Right
        self.action_space = gym.spaces.Discrete(5)
        #Assign to each agent the possibility to move: Stay, Up, Down, Left, Right      
        print("finished init successfully")
                
    def reset(self):
        
        #Reset observations for each agent once reset is called
        obs = {}

        ########### NEW ############
        #Reset obs spaces for each agent once reset method is called
        #Different strategies and possibilities on how to handle the reset!
        #Options:
        #1) Reset also all target nodes locations
        """
        If so I need to change the graph class
        """
        #2) Each reset also randomizes the agents starting locations for each episodes
        
        #Assign new random starting location to agents as a dictionary
        self._agents_locations = self._set_agents_random_starting_locations(self._graph)
        #print(self._agents_locations)
        
        #3) Use the same agents starting locations every reset -> risk of overfitting?
        """
        """
            
        for i, agent in enumerate(self.agents):
            current_pos_and_target_locations_tuples = [self._agents_locations[agent]]
            for j, target in enumerate(self._target_nodes_locations):
                current_pos_and_target_locations_tuples.append(self._target_nodes_locations[target])
            obs[agent] = {"obs": np.array(current_pos_and_target_locations_tuples).flatten()}
        
        print(obs)
        
        return obs
                
    def step(self, action_dict):
        
        obs = {}
        rewards = {}
        done = {"__all__": False}
        info = {}
        
        #Key, value in dict.items()
        for agent_id, action in action_dict.items():
            
            # Get the current agent's node location            
            agent_starting_node = self._agents_locations[agent_id]
            
            new_agent_location = self._move_agent(agent_starting_node, action, self._graph)
                        
            # Check if the current agent is on a target node
            self._agents_locations[agent_id] = new_agent_location            
            
            if new_agent_location in self._target_nodes_locations:
                # Give reward of 1 if agent is on a target node
                rewards[agent_id] = 1
                done[agent_id] = False
            else:
                rewards[agent_id] = 0
                done[agent_id] = False
                
            # Update observation dictionary for this agent -> update only his position since targets are static
            #print(new_agent_location)            
            current_pos_and_target_locations_tuples = [self._agents_locations[agent_id]]
            for j, target in enumerate(self._target_nodes_locations):
                current_pos_and_target_locations_tuples.append(self._target_nodes_locations[target])
            obs[agent_id] = {"obs": np.array(current_pos_and_target_locations_tuples).flatten()}
        
        return obs, rewards, done, info
    
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
        #Collect target nodes locations
        return {node: node for node in xnetwork_graph.G.nodes if xnetwork_graph.G.nodes[node]['color'] == 'red'}
    
    def _set_agents_random_starting_locations(self, networkx_graph):
        #Generate and assign random starting positions to agents
        return {agent_ID: random.choice(list(networkx_graph.G.nodes)) for agent_ID in self.agents}
    
    #Try to move agent
    def _move_agent(self, agent_starting_node, action, graph):
        
        movements = {
            0: (0, 0),
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1)
        }
                
        move = movements[action]
        new_agent_location = (agent_starting_node[0] + move[0], agent_starting_node[1] + move[1])
                
        if graph.G.has_edge(agent_starting_node, new_agent_location):
            agent_starting_node = new_agent_location
                
        return agent_starting_node
    
    """
    def _old_step_function(self):

        obs = {}
        rewards = {}
        done = {"__all__": False}
        info = {}
        
        #Key, value in dict.items()
        for agent_id, action in action_dict.items():
            
            # Get the current agent's node location            
            agent_starting_node = self._agents_locations[agent_id]
            new_agent_location = agent_starting_node
            
            #Try moving the agent and if succesfull save in new agent location the new position
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
                        
            # Check if the current agent is on a target node
            self._agents_locations[agent_id] = new_agent_location            
            
            if new_agent_location in self._target_nodes_locations:
                # Give reward of 1 if agent is on a target node
                rewards[agent_id] = 1
                done[agent_id] = False
            else:
                rewards[agent_id] = 0
                done[agent_id] = False
                
            # Update observation dictionary for this agent -> update only his position since targets are static
            #print(new_agent_location)            
            current_pos_and_target_locations_tuples = [self._agents_locations[agent_id]]
            for j, target in enumerate(self._target_nodes_locations):
                current_pos_and_target_locations_tuples.append(self._target_nodes_locations[target])
            obs[agent_id] = {"obs": np.array(current_pos_and_target_locations_tuples).flatten()}
        
        return obs, rewards, done, info
    """

#Testing

fake_env_config = {
    "size": 10,
    "num_agents": 1,
    "num_patrollers": 1,
    "num_attackers": 0
}

fake_env_config_2 = {
    "size": 10,
    "num_agents": 2,
    "num_patrollers": 1,
    "num_attackers": 1
}

grafo = RayGraphEnv(fake_env_config_2)
grafo.reset()
obs, rew, done, info = grafo.step({"Patroller_0": 2, "Attacker_0": 1})
#print(obs)
#print(rew)
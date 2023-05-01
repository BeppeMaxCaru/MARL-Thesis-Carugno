import ray
import gym as gym
import numpy as np
import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import graph

#Policy_mapping_dict deve essere configurato correttamente con alcune opzioni
#Se si usa come "all_scenario" non serve configurazione con map_name
#siccome verrà sempre usata all_scenario altrimenti va fornito uno scenario con map_name

#Ex.
"""
policy_mapping_dict = {
    "all_scenario": {
        "description": "patrolling_test",
        "team_prefix": ("Patroller_", "Attacker_"),
        "all_agents_one_policy": False, #Defines if agent have a shared policy or each one has its own
        "one_agent_one_policy": True, #Defines if each agent should have its own policy or not
    },
}
"""

# vs

# Dove patrolling_graph è il "map_name"

policy_mapping_dict = {
    "patrolling_graph": {
        "description": "only patrollers",
        #"team_prefix": ("Patroller_", "Attacker_"),
        "team_prefix": ("Patroller_",),
        
        #Wheter defining here "all_agents_one_policy" and/or "one_agent_one_policy" depends
        #on the share_policy parameter used in [algo].fit(...)
        # 1) If share_policy == "all" "all_agents_one_policy" is used and has to be set to True
        # 2) If share_policy == "group" "all_agents_one_policy" is used, it has to be set to True
        # and "team_prefix" is used to share the policy but only between members of the same groups
        # 3) If share_policy == "individual" "one_agent_one_policy" is used and has to be set True
        
        "all_agents_one_policy": True, #Defines if agent have a shared policy or each one has its own
        "one_agent_one_policy": True, #Defines if each agent should have its own policy or not
    }
}

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
        self.agents = ["Patroller_{}".format(i) for i in range(self.num_patrollers)]
        #Add attackers
        #self.agents = self.agents + ["Attacker_{}".format(i) for i in range(self.num_attackers)]
        
        self.num_agents = len(self.agents)
        
        #Generate new graph        
        self._graph = self._generate_new_graph(self._graph_size)
        #Collect new target nodes locations from graph as a dictionary
        self._target_nodes_locations = self._get_target_nodes_locations(self._graph)
        
        #Since I cannot use a nested dict in obs so I flatten it to a big Box
        #Max values are minimum and maximum graph size with shape as explained below
        self.observation_space = gym.spaces.Dict({
            #shape = (agents positions and targets  positions) * 2 since both are tuples of 2 elems
            "obs": gym.spaces.Box(low=0, high=self._graph_size, shape=(2 + len(self._target_nodes_locations)*2,), dtype=np.int32),
            #"state": gym.spaces.Box(low=0, high=self._graph_size, shape=(len(self._target_nodes_locations)*2,), dtype=np.int32),
            #"opponent_obs": None,
        })
        
        self.iter_counter = 0
        
        #Same 5 actions for all patrollers and attackers: Stay, Up, Down, Left, Right
        self.action_space = gym.spaces.Discrete(5)
        
        print(self.observation_space)
        print(self.action_space)
        print(self.num_agents)
        print(self.agents)

        self.env_config = env_config
                        
    def reset(self):
        
        self.iter_counter = 0
        
        #Reset observations for each agent once reset is called
        obs = {}
        
        #Assign new random starting location to agents as a dictionary
        self._agents_locations = self._set_agents_random_starting_locations(self._graph)
        #print(self._agents_locations)
        
        for i, agent in enumerate(self.agents):
            current_pos_and_target_locations_tuples = [self._agents_locations[agent]]
            for j, target in enumerate(self._target_nodes_locations):
                current_pos_and_target_locations_tuples.append(self._target_nodes_locations[target])
            obs[agent] = {"obs": np.array(current_pos_and_target_locations_tuples, dtype=np.int32).flatten()}
        print(obs)
        
        return obs
                
    def step(self, action_dict):
        
        obs = {}
        rewards = {}
        dones = {"__all__": False}
        info = {}
        
        self.iter_counter = self.iter_counter + 1
        print("timestep number: " + str(self.iter_counter))
                
        #Key, value in dict.items()
        for agent_id, action in action_dict.items():
            
            # Get the current agent's node location            
            agent_starting_node = self._agents_locations[agent_id]
            new_agent_location = self._move_agent(agent_starting_node, action, self._graph)
                        
            # Check if the current agent is on a target node
            self._agents_locations[agent_id] = new_agent_location
                
            # Give reward of 1 if agent is on a target node else 0
            rewards[agent_id] = 1 if new_agent_location in self._target_nodes_locations else 0
            dones[agent_id] = False if self.iter_counter < 5 else True
            #dones[agent_id] = True if new_agent_location in self._target_nodes_locations else False
            
            # Update observation dictionary for this agent -> update only his position since targets are static
            #print(new_agent_location)            
            current_pos_and_target_locations_tuples = [self._agents_locations[agent_id]]
            for j, target in enumerate(self._target_nodes_locations):
                current_pos_and_target_locations_tuples.append(self._target_nodes_locations[target])
            obs[agent_id] = {"obs": np.array(current_pos_and_target_locations_tuples).flatten()}
        
        #print(action_dict)
        #print(obs)
        
        return obs, rewards, dones, {} #info
    
    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env_config["max_steps"],
            "policy_mapping_info": policy_mapping_dict
        }
        print(env_info)
        return env_info
    
    def close(self):
        return
    
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
        #If Stay return starting node
        if move == 0: 
            return agent_starting_node
        #Otherwise calculate new location
        new_agent_location = (agent_starting_node[0] + move[0], agent_starting_node[1] + move[1])
        #Check if new location exists in graph and if yes return it
        if graph.G.has_edge(agent_starting_node, new_agent_location):
            return new_agent_location
        #If not return starting node        
        return agent_starting_node

#Testing
#NB When testing always comment the below code
"""
fake_env_config_1 = {
    "size": 10,
    "num_agents": 2,
    "num_patrollers": 2,
    "num_attackers": 0
}

fake_env_config_2 = {
    "size": 10,
    "num_agents": 2,
    "num_patrollers": 1,
    "num_attackers": 1
}

grafo = RayGraphEnv(fake_env_config_1)
grafo.reset()
obs, rew, dones, info = grafo.step({"Patroller_0": 2, "Patroller_1": 1})
#print(obs)
#print(rew)

for i in range(10):
    obs, rew, dones, info = grafo.step({"Patroller_0": random.randint(0, 4), "Patroller_1": random.randint(0, 4)})
    print(obs)
    print(rew)

print(grafo.get_env_info())
"""

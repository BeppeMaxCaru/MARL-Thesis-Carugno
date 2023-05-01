#NB Important to read of Parallel API!
import pettingzoo
import gym
import random
import numpy as np

from pettingzoo.test import parallel_api_test

import graph

class PettingZooGraph(pettingzoo.ParallelEnv):
    metadata = {"render_modes": ["human"], "render_fps": 15}
    
    def __init__(self, env_config):
        
        #Set graph size
        self._graph_size = env_config["size"]
        
        #Collect list of all possible agents
        self.possible_agents = ["Patroller_" + str(i) for i in range(env_config["num_patrollers"])]
        self.possible_agents = self.possible_agents + ["Attacker_" + str(i) for i in range(env_config["num_attackers"])]
        
        #Map agents to indices
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        #Declare action space for each agent 
        #NB Agents in pettingzoo can have different observation spaces
        #which is not possible in RLLIB
        self.action_spaces = {agent: gym.spaces.Discrete(5) for agent in self.possible_agents}
        
        #Generate graph
        self._graph = self._generate_new_graph(self._graph_size)
        #Get target nodes locations (is a dictionary)
        self._target_nodes_locations = self._get_target_nodes_locations(self._graph)
        #Get agents starting random locations
        self._agents_location = self._set_agents_random_starting_locations(self._graph, self.possible_agents)
        
        #Set observation spaces
        self.observation_spaces = {agent: 
            gym.spaces.Box(low=0, high=self._graph_size, shape=(2 + len(self._target_nodes_locations)*2,), dtype=np.int32)
            for agent in self.possible_agents
            }
        
        #self.render_mode = render_mode
        
    def render(self):
        pass
        
    def close(self): 
        pass
    
    def reset(self):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.observations = {agent: None for agent in self.agents}
        
    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}        
        
        observations = {}
        rewards = {}
        
        #Define obs and rewards for each agent
        for agent in self.agents:
        
            agent_starting_location = self._agents_location[agent]
            new_agent_location = self._move_agent(agent_starting_location, action, self._graph)        
            
            observations[agent] = [new_agent_location[0], new_agent_location[1]]
            for node in self._target_nodes_locations:
                observations[agent] = observations[agent] + [node[0], node[1]]            
            
            # rewards for all agents are placed in the rewards dictionary to be returned
            if (new_agent_location in self._target_nodes_locations):
                rewards[agent] = 1
            else:
                rewards[agent] = 0
                
            self._agents_location[agent] = new_agent_location
        
        #We don't need this since agents do not die or terminate
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
        
    def _generate_new_graph(self, side_dim_of_squared_graph):
        #Generate new graph
        return graph.Graph(side_dim_of_squared_graph, side_dim_of_squared_graph)
    
    def _get_target_nodes_locations(self, xnetwork_graph):
        #Collect target nodes locations
        return {node: node for node in xnetwork_graph.G.nodes if xnetwork_graph.G.nodes[node]['color'] == 'red'}
    
    def _set_agents_random_starting_locations(self, networkx_graph, agents):
        #Generate and assign random starting positions to agents
        return {agent: random.choice(list(networkx_graph.G.nodes)) for agent in agents}
    
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
    
config = {
    "size": 10,
    "num_patrollers": 1,
    "num_attackers": 0
}
PettingZooGraph(config)

parallel_api_test(PettingZooGraph(config), num_cycles=10)
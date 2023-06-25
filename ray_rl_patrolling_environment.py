import ray
import gym as gym
import numpy as np
import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import networkx as nx
import matplotlib.pyplot as plt

import patrolling_graph

#Policy_mapping_dict deve essere configurato correttamente con alcune opzioni
#Se si usa come "all_scenario" non serve configurazione con map_name
#siccome verrà sempre usata all_scenario altrimenti va fornito uno scenario con map_name

#Ex.

policy_mapping_dict = {
    "all_scenario": {
        "description": "patrolling_all_scenario",
        "team_prefix": ("Patroller_",), # OR ("Patroller_", "Attacker_") if also attackers != 0
        
        #Wheter defining here "all_agents_one_policy" and/or "one_agent_one_policy" depends
        #on the share_policy parameter used in [algo].fit(...)
        # 1) If share_policy == "all" "all_agents_one_policy" is used and has to be set to True
        # 2) If share_policy == "group" "all_agents_one_policy" is used, it has to be set to True
        # and "team_prefix" is used to share the policy but only between members of the same groups
        # 3) If share_policy == "individual" "one_agent_one_policy" is used and has to be set True
        
        "all_agents_one_policy": True, #Defines if agent have a shared policy or each one has its own
        "one_agent_one_policy": True, #Defines if each agent should have its own policy or not
    },
}

# vs
"""
# Dove only_patrollers è il "map_name"
policy_mapping_dict = {
    "only_patrollers": {
        "description": "patrollers",
        "team_prefix": ("Patroller_",),
        "all_agents_one_policy": True, #Defines if agent have a shared policy or each one has its own
        "one_agent_one_policy": True, #Defines if each agent should have its own policy or not
    },
    "patrollers_and_attackers": {
        "description": "patrollers and attackers",
        "team_prefix": ("Patroller_", "Attacker_",),
        "all_agents_one_policy": True, #Defines if agent have a shared policy or each one has its own
        "one_agent_one_policy": True, #Defines if each agent should have its own policy or not
    }
}
"""

class RayGraphEnv(MultiAgentEnv):
    
    #env_config is a configuration dictionary containing the parameters to use to setup the env
    def __init__(self, env_config):
        
        #Configuration file settings + check
        #Define size of the squared grid        
        self._graph_size = env_config["size"]
        # Define the possible agents in the environment
        self.num_patrollers = env_config["num_patrollers"]
        self.num_attackers = env_config["num_attackers"]
        
        #Seed to initialize and control random number generators and ensure
        #reproducibility of experiments
        self.seed_for_random_numbers_generators = int(env_config["seed_for_random_numbers_generators"])
        random.seed(self.seed_for_random_numbers_generators)
        np.random.seed(self.seed_for_random_numbers_generators)
        
        self.max_steps_per_episode = env_config["episode_limit"]
                
        #Add patrollers
        self.agents = ["Patroller_{}".format(i) for i in range(self.num_patrollers)]
        #Add attackers
        self.agents = self.agents + ["Attacker_{}".format(i) for i in range(self.num_attackers)]
                
        self.num_agents = self.num_patrollers + self.num_attackers
        
        #Generate new graph        
        self._graph = self._generate_new_graph(self._graph_size)
        #Collect new target nodes locations from graph as a dictionary
        #Collect them here since they're static so they do not change in resets
        self._target_nodes_locations = self._get_target_nodes_locations(self._graph)
        
        #Used to create more complex reward functions
        self._last_visited_target_nodes_by_agents = {} #Dict agent to last visited nodes from him
        self._visited_target_nodes_by_group = [] #list with visited target from group o agents
        self._target_nodes_priority_queue = []
        
        #Since I cannot use a nested dict in obs so I flatten it to a big Box
        #Max values are minimum and maximum graph size with shape as explained below
        self.observation_space = gym.spaces.Dict({
            #shape = (agents positions and targets  positions) * 2 since both are tuples of 2 elems
            #"obs": gym.spaces.Box(low=0, high=self._graph_size, shape=(2 + len(self._target_nodes_locations)*2,), dtype=np.int32),
            
            #Obs space with just agent position -> this is the correct one!
            "obs": gym.spaces.Box(low=0, high=self._graph_size, shape=(2,), dtype=np.int32),

            #"state": gym.spaces.Box(low=0, high=self._graph_size, shape=(len(self._target_nodes_locations)*2,), dtype=np.int32),
            #"opponent_obs": None,
        })
        #Cerca di usare "state" per passare in modo più semplice posizione dei nodi target? Non serve posizione nodi siccome sono statici
        
        #Used to interrupt episodes when limit timesteps are reached
        self.current_episode_timesteps_counter = 0
        
        self.current_episode_number = 0
        self.total_timesteps_counter = 0
        
        #Same 5 actions for all patrollers and attackers: Stay, Up, Down, Left, Right
        #self.action_space = gym.spaces.Discrete(5)
        #Make it continuous but discretized through the dtype -> this way IDDPG works!
        self.action_space = gym.spaces.Box(low=0, high=4, shape=(), dtype=np.int32)
                
        self.env_config = env_config
                                
    def reset(self):

        self.current_episode_timesteps_counter = 0
        
        #Reset observations for each agent once reset is called
        obs = {}
        
        #Assign new random starting location to agents as a dictionary
        self._agents_locations = self._set_agents_random_starting_locations(self._graph)
        
        #Set array of last visited nodes by agents
        self._last_visited_target_nodes_by_agents = {}
        for i, agent in enumerate(self.agents):
            if self._agents_locations[agent] in self._target_nodes_locations:
                self._last_visited_target_nodes_by_agents[agent] = self._agents_locations[agent]
            
        #Reset queue and fill it
        self._target_nodes_priority_queue = []
        for target in self._target_nodes_locations:
            self._target_nodes_priority_queue.append(target)
        
        for i, agent in enumerate(self.agents):
            current_pos_and_target_locations_tuples = [self._agents_locations[agent]]
            #for j, target in enumerate(self._target_nodes_locations):
                #current_pos_and_target_locations_tuples.append(self._target_nodes_locations[target])
            obs[agent] = {"obs": np.array(current_pos_and_target_locations_tuples, dtype=np.int32).flatten()}
                
        return obs
                
    def step(self, action_dict):
        
        obs = {}
        rewards = {}
        info = {}
        
        done_flag = False
        
        #Key, value in dict.items()
        for agent_id, action in action_dict.items():
                        
            # Get the current agent's node location            
            agent_starting_node = self._agents_locations[agent_id]
            new_agent_location = self._move_agent(agent_starting_node, action, self._graph)
                        
            self._agents_locations[agent_id] = new_agent_location
            
            ######### REWARD FUNCTIONS PART #################################
                
            # Give reward of 1 if agent is on a target node else 0
            rewards[agent_id] = self._one_if_agent_on_target_node_else_zero(agent_id, new_agent_location)
            
            #Give 1 if agent on target node and target node is different than the previously visited one by the agent
            #rewards[agent_id] = self._one_if_agent_on_new_target_node_else_zero(agent_id, new_agent_location)
            
            #Give 1 if the target node is the first one in the queue so the last visited
            #rewards[agent_id] = self._minimize_max_idleness(agent_id, new_agent_location)

            #Give 1 if the node visited is target, is different than previous target node
            #and there's not another patroller on it?
            
            ########## END REWARD FUNCTIONS ##################################

            # Update observation dictionary for this agent -> update only his position since targets are static
            #print(new_agent_location)            
            current_pos_and_target_locations_tuples = [self._agents_locations[agent_id]]
            #Comment the for loop to use just the agent location in the observation space
            #for j, target in enumerate(self._target_nodes_locations):
                #current_pos_and_target_locations_tuples.append(self._target_nodes_locations[target])
            obs[agent_id] = {"obs": np.array(current_pos_and_target_locations_tuples).flatten()}
            
        ########### Va definito dones altrimenti simulazione non si fermerà mai!!!!!! ################
        
        #If episode limit not reached increase timesteps counters
        if (self.current_episode_timesteps_counter < self.max_steps_per_episode):
            self.current_episode_timesteps_counter = self.current_episode_timesteps_counter + 1
            self.total_timesteps_counter = self.total_timesteps_counter + 1
        #If episode limit reached increase episode counter and set done_flag to True to call reset()
        if (self.current_episode_timesteps_counter >= self.max_steps_per_episode):
            done_flag = True
            #Counters
            self.current_episode_number = self.current_episode_number + 1
        dones = {"__all__": done_flag}
        #NB if dones __all__ diventa true current_episode_viene_resettato a 0 siccome
        # viene chiamata reset
        
        if self.env_config["enable_rendering"]:
            self.render()
        
        return obs, rewards, dones, info
    
    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_steps_per_episode,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
    
    def close(self):
        return
    
    def render(self):
        
        if not hasattr(self, "fig"):
            # Create the plot figure and axis only once
            self.fig, self.ax = plt.subplots()
        
        original_nodes_colours = {}
        #Color nodes of agents differently only for this function call
        for agent, node in self._agents_locations.items():
            original_nodes_colours[node] = self._graph.G.nodes[node]['color']
            if "Patroller_" in agent:
                self._graph.G.nodes[node]['color'] = 'blue'
            elif "Attacker_" in agent:
                self._graph.G.nodes[node]['color'] = 'red'
        
        #Clear previous plot
        self.ax.clear()
                
        #Create plot
        nx.draw_networkx(self._graph.G,
                         pos=self._graph.pos, 
                         arrows=None, 
                         with_labels=False,
                         node_size=100,
                         node_color=[self._graph.G.nodes[node]['color'] for node in self._graph.G.nodes()])
        
        plt.suptitle("Timestep number of current episode: " + str(self.current_episode_timesteps_counter))
        plt.title("Episode number " + str(self.current_episode_number))
    
        plt.draw()
        plt.pause(0.25)
        
        #Reset colours
        for node, color in original_nodes_colours.items():
            #This if fixes cases in which two agents on same node so colour is reset correctly
            self._graph.G.nodes[node]['color'] = 'green' if node in self._target_nodes_locations else 'black'

        return
        
    ############## PRIVATE FUNCTIONS #############
        
    def _generate_new_graph(self, side_dim_of_squared_graph):
        #Generate new graph
        return patrolling_graph.Graph(side_dim_of_squared_graph, side_dim_of_squared_graph)
    
    def _get_target_nodes_locations(self, xnetwork_graph):
        #Collect target nodes locations
        return {node: node for node in xnetwork_graph.G.nodes if xnetwork_graph.G.nodes[node]['color'] == 'green'}
    
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
    
    ############## REWARDS FUNCTIONS #############
    def _one_if_agent_on_target_node_else_zero(self, agent_id, new_agent_location):
        reward = 0
        #return 1 if agent_location in self._target_nodes_locations else 0
        reward = 1 if new_agent_location in self._target_nodes_locations else 0
        return reward
    
    def _one_if_agent_on_new_target_node_else_zero(self, agent_id, new_agent_location):
        reward = 0
        reward = 1 if (   new_agent_location in self._target_nodes_locations
                     and
                        self._last_visited_target_nodes_by_agents.get(agent_id) is not None 
                     and
                        self._last_visited_target_nodes_by_agents.get(agent_id) is not new_agent_location) else 0
        #Update the most recently visited target node
        if new_agent_location in self._target_nodes_locations:
            self._last_visited_target_nodes_by_agents[agent_id] = new_agent_location
        #Return reward    
        return reward

    def _one_if_agent_on_new_target_for_all_group_else_zero(self, agent_id, new_agent_location):
        reward = 0
        if new_agent_location in self._target_nodes_locations:
            #Check if target already visited by group
            if new_agent_location not in self._visited_target_nodes_by_group:
                reward = 1
                self._visited_target_nodes_by_group.append(new_agent_location)
        #If all targets already visited by the group empty visited targets list and restart populating it
        if set(self._target_nodes_locations) == set(self._visited_target_nodes_by_group):
            self._visited_target_nodes_by_group = []
        return reward
        
    def _minimize_max_idleness(self, agent_id, new_agent_location):
        #Use priority queue to track highest priority target nodes
        reward = 0
        #Agent is on highest priority node
        if new_agent_location == self._target_nodes_priority_queue[0]:
            reward = 1
        #If node is target but not high priority one put it at the back of queue with no reward
        if new_agent_location in self._target_nodes_locations:
            #Update queue by adding node at the back
            self._target_nodes_priority_queue.append(self._target_nodes_priority_queue[0])
            self._target_nodes_priority_queue.pop(0)
        return reward
    
    def _minimize_average_idleness(self, agent_id, new_agent_location):
        #Use priority queue to track highest priority target nodes
        reward = 0
        #Agent is on target node with priority higher than average
        if new_agent_location in self._target_nodes_priority_queue:
            node_position_in_queue = self._target_nodes_priority_queue.index(new_agent_location)
            if (node_position_in_queue < (len(self._target_nodes_priority_queue) / 2)):
                #Give reward if target is in starting half of the queue
                reward = 1
            #Update queue by adding the visited target node at the back but reward only if starting half of the queue
            self._target_nodes_priority_queue.append(self._target_nodes_priority_queue[node_position_in_queue])
            self._target_nodes_priority_queue.pop(node_position_in_queue)
        return reward
    
class RayGraphEnv_Coop(RayGraphEnv):
    
    def step(self, action_dict):
        
        obs = {}
        rewards = {}
        info = {}
        
        done_flag = False
        
        #Key, value in dict.items()
        for agent_id, action in action_dict.items():
                        
            # Get the current agent's node location            
            agent_starting_node = self._agents_locations[agent_id]
            new_agent_location = self._move_agent(agent_starting_node, action, self._graph)
                        
            # Check if the current agent is on a target node
            self._agents_locations[agent_id] = new_agent_location
            
            ############# REWARD FUNCTIONS ######################
            
            # Give reward of 1 if agent is on a target node else 0
            #rewards[agent_id] = self._one_if_agent_on_target_node_else_zero(agent_id, new_agent_location)
            
            #Give 1 if agent on target node and target node is different than the previously visited one
            rewards[agent_id] = self._one_if_agent_on_new_target_node_else_zero(agent_id, new_agent_location)
            
            #Give 1 if target still not visited by other agents of same team else 0
            #rewards[agent_id] = self._one_if_agent_on_new_target_for_all_group_else_zero(agent_id, new_agent_location)

            #Give 1 if the target node is the first one in the queue so the last visited
            #rewards[agent_id] = self._minimize_max_idleness(agent_id, new_agent_location)
            
            #Give 1 if the target node is in first half of the queue so not one of the most visited recently
            #rewards[agent_id] = self._minimize_average_idleness(agent_id, new_agent_location)
            
            ############# REWARD FUNCTIONS SECTION END ######################
            
            # Update observation dictionary for this agent -> update only his position since targets are static
            #print(new_agent_location)           
            current_pos_and_target_locations_tuples = [self._agents_locations[agent_id]]
            #Comment the for loop to use just the agent location in the observation space
            #for j, target in enumerate(self._target_nodes_locations):
                #current_pos_and_target_locations_tuples.append(self._target_nodes_locations[target])
            obs[agent_id] = {"obs": np.array(current_pos_and_target_locations_tuples).flatten()}
            
        #Split the total reward equally among agents to make them work together
        total_reward = 0
        #Cumulate rewards to get the total
        for agent_id, reward in sorted(rewards.items()):
            total_reward = total_reward + reward
        #Update the reward values in the rewards dictionary making them equal
        for agent_id, reward in sorted(rewards.items()):
            rewards[agent_id] = total_reward / self.num_agents
        
        ########### Va definito dones altrimenti simulazione non si fermerà mai!!!!!! ################
        
        self.current_episode_timesteps_counter = self.current_episode_timesteps_counter + 1
        if (self.current_episode_timesteps_counter >= self.max_steps_per_episode):
            done_flag = True
        dones = {"__all__": done_flag}
        
        if self.env_config["enable_rendering"]:
            self.render()
        
        return obs, rewards, dones, info
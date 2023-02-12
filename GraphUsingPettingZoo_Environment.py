import gymnasium as gym
import pettingzoo
import graph
import numpy as np
import random

import functools

from gym.utils.env_checker import check_env

import stable_baselines3 as sb3

from stable_baselines3.common.env_checker import check_env

from pettingzoo.test import api_test
from pettingzoo.test import parallel_api_test
from pettingzoo.utils import agent_selector

import ray
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms import ppo

#Not paraller version
class PettingZooGraphEnv(pettingzoo.AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 15}
    
    def __init__(self, render_mode=None, size=10, num_patrollers=1, num_attackers=0):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        
        self.graph_size = size
        
        
        
        #Define the possible agents in the environment
        self.num_patrollers = num_patrollers
        self.num_attackers = num_attackers
        
        #PettingZoo requires that the agents are defined in a list
        self.possible_agents = []
        #Add the patrollers to the list
        for i in range(self.num_patrollers):
            self.possible_agents.append("patroller_" + str(i))
        #Add the attackers to the list
        for i in range(self.num_attackers):
            self.possible_agents.append("attacker_" + str(i))
        
        #Map the agent names to their index in the list
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
                
        #Observation spaes
        self.observation_spaces = {
            agent: gym.spaces.Dict() for agent in self.possible_agents
        }
        
        #self._setup_new_episode()
        
        for agent in self.possible_agents:
            self.observation_spaces[agent]['current_agent_node_location'] = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.graph_size, self.graph_size]), dtype=np.int32)
        
        #Action space is the number of possible moves from a node: "Stay", "Up", "Down", "Left", "Right"
        #Same for both patrollers and attackers
        self.action_spaces = {agent: gym.spaces.Discrete(5) for agent in self.possible_agents}
        
        self.render_mode = None
    
    
    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self.observation_spaces[agent]
        
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    #New using pettingzoo
    #observe() is used to get the observation of the specified agent
    def observe(self, agent):
        
        #Observe should return the observation of the specified agent. This function
        #should return a sane observation (though not necessarily the most up to date possible)
        #at any time after reset() is called.
        
        observation = {}
        observation['current_agent_node_location'] = np.array([self.agents_locations[agent][0], self.agents_locations[agent][1]])
        for i in range(len(self.target_nodes)):
            observation['target_node_' + str(i) + '_position'] = np.array([self.target_nodes_positions_dict[self.target_nodes[i]][0], self.target_nodes_positions_dict[self.target_nodes[i]][1]])
        
        return observation
    
    def _get_info(self):
        pass
    
    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.setup_new_graph()       
                   
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
            
    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        
        #Get the current agent
        agent = self.agent_selection
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        last(observe=True) returns observation, reward, done, and info for the agent 
        currently able to act. The returned reward is the cumulative reward that the 
        agent has received since it last acted. If observe is set to False, the 
        observation will not be computed, and None will be returned in its place. 
        Note that a single agent being done does not imply the environment is done.
        """
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
        #Get the current agent location and the previous location
        self.current_agent_node = self.agents_locations[agent]
        self.previous_agent_node = self.current_agent_node
                
        #Move the current agent
        if action == 0:
            #Stay in the same node
            pass
        elif action == 1:
            #Move to the node above in the graph if edge going to it exists
            #Check if the node above exists and if there is an edge going to it
            if (self.current_agent_node[0] > 0) and (self.graph.G.has_edge(self.current_agent_node, (self.current_agent_node[0]-1, self.current_agent_node[1]))):
                #Move to the node above
                self.current_agent_node = (self.current_agent_node[0]-1, self.current_agent_node[1])
        elif action == 2:
            #Move to the node down in the graph if edge exists
            #Check if the node below exists and if there is an edge going to it
            if (self.current_agent_node[0] < self.graph.n-1) and (self.graph.G.has_edge(self.current_agent_node, (self.current_agent_node[0]+1, self.current_agent_node[1]))):
                #Move to the node below
                self.current_agent_node = (self.current_agent_node[0]+1, self.current_agent_node[1])
        elif action == 3:
            #Move to the node on the left in the graph if edge exists
            #Check if the node to the left exists and if there is an edge going to it
            if (self.current_agent_node[1] > 0) and (self.graph.G.has_edge(self.current_agent_node, (self.current_agent_node[0], self.current_agent_node[1]-1))):
                #Move to the node to the left
                self.current_agent_node = (self.current_agent_node[0], self.current_agent_node[1]-1)
        elif action == 4:
            #Move to the node on the right in the graph if edge exists
            #Check if the node to the right exists and if there is an edge going to it
            if (self.current_agent_node[1] < self.graph.m-1) and (self.graph.G.has_edge(self.current_agent_node, (self.current_agent_node[0], self.current_agent_node[1]+1))):
                #Move to the node to the right
                self.current_agent_node = (self.current_agent_node[0], self.current_agent_node[1]+1)
        else:
            #Invalid action
            raise ValueError("Invalid action: {}".format(action))
        
        #Check if the agent is on a target node and if it is reducr its idleness to 0
        node_color = self.graph.G.nodes[self.current_agent_node]['color']
        if node_color == 'r':
            #If the agent is on a red node, reset its idleness to 0
            self.target_nodes_idleness[self.current_agent_node] = 0
        
        #Simple reward function for testing
        #0 if agent is not on a target node, 1 if it is but it is not the same node as the previous step
        reward = 0 if self.current_agent_node not in self.target_nodes else 1 if self.current_agent_node != self.previous_agent_node else 0
        self.previous_agent_node = self.current_agent_node
        #Update the integer reward for the current agent during this pass of agents actions
        #self.rewards = {agent: 0 for agent in self.agents}
        self.rewards[agent] = self.rewards.get(agent, 0) + reward
        #self.rewards[agent] = reward        
        
        #Update cumulative rewards if it is the last agent of the current pass of agents actions
        self._accumulate_rewards()
        
        print("Agents: {}, Reward: {}".format(self.agents, reward))
        print("Agents: {}, Reward dict values: {}".format(self.agents, self.rewards))
        print("Agents: {}, Cumulative rewards: {}".format(self.agents, self._cumulative_rewards))
        
        #Update agent locations
        self.agents_locations[agent] = self.current_agent_node
        self.agents_previous_locations[agent] = self.previous_agent_node
        
        #Update cumulative rewards if it is the last agent of the current pass of agents actions
        if self._agent_selector.is_last():
            
            #If the agent is the last one to act increase the target nodes idleness before the new agents actions round
            #Increase idleness of all target nodes by 1
            self.target_nodes_idleness = {node: idleness + 1 for node, idleness in self.target_nodes_idleness.items()}
                        
            #Clear rewards dictionary for the next pass of agents actions
            #self._clear_rewards()
            #After last agent has acted, update rewards and cumulative rewards dictionary
            #self.rewards = {agent: 0 for agent in self.agents}
            #self._cumulative_rewards = {agent: 0 for agent in self.agents}
        """
        else:
            self._clear_rewards()
        """
            
        #Select the next agent
        self.agent_selection = self._agent_selector.next()
        
    #Pettingzoo version of the setup_new_episode function
    def setup_new_graph(self):
        #Per ogni episodio:
        #0) Reset total idleness
        #1) Un nuovo grafo
        #2) Un nuovo nodo di partenza per ogni agente
        #3) Una nuova lista di nodi target condivisa
        #4) Observation space aggiornato per ciascun agente
        
        #Reset total idleness
        self.total_idleness = 0
        
        #Generate new graph
        self.graph = graph.Graph(self.graph_size, self.graph_size)
        
        #Agents all alive at the beginning of each episode
        self.agents = self.possible_agents
        
        self.agents_locations = {}
        self.agents_previous_locations = {}
        
        #Assign to each agent a random node as the starting location
        for agent in self.agents:
            self.agents_locations[agent] = random.choice(list(self.graph.G.nodes))
            #Store also the agents previous node location
            self.agents_previous_locations[agent] = self.agents_locations[agent]
        
        #Retrieve new list of target nodes (nodes with red color)
        self.target_nodes = [node for node in self.graph.G.nodes if self.graph.G.nodes[node]['color'] == 'red']
        #Determine the position of each target node and assign it to a dictionary for gym observation space
        self.target_nodes_positions_dict = {node: self.graph.G.nodes[node]['position'] for node in self.target_nodes}
        
        #Set target nodes idleness to 0
        self.target_nodes_idleness = {node: 0 for node in self.target_nodes}
        
        #Create dictionary to convert node to index
        self.node_to_index = {node: index for index, node in enumerate(self.graph.G.nodes)}    
        
        for i in range(len(self.target_nodes)):
            self.observation_spaces[agent]['target_node_' + str(i) + '_position'] = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.graph_size, self.graph_size]), dtype=np.int32)
        
    def render(self, mode='human'):
        #During the drawing of the graph, the agent's current node is highlighted in green
        self.graph.G.nodes[self.current_agent_node]['color'] = 'green'
        self.graph.draw()
        #pass
        
    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    #def move_agent(action):
        

#########################
#Class testing

env = PettingZooGraphEnv()
#env.graph.draw()

env.reset()

#print(env.possible_agents)
#print(env.agents)
#print(env.observation_spaces)
#print(env.action_spaces)

for i in range(10):
    api_test(env, num_cycles=10, verbose_progress=True)

print("Starting test of the environment...")

"""
#Test the environment
for i in range(25):
    #Current agent position
    current_agent_position = env.agents_locations[env.agent_selection]
    print("Current agent position: {}".format(current_agent_position))
    env.step(action=env.action_spaces[env.agent_selection].sample())
    #Step the environment and print the resulting action, observation, reward and done)
    current_agent = env.agent_selection
    print("Current agent: {}".format(current_agent))
    #Action selected by agent
    action = env.action_spaces[current_agent].sample()
    print("Action: {}".format(action))
    #Observation made by agent
    observation = env.observe(current_agent)
    print("Observation: {}".format(observation))
    #Reward obtained by agent
    reward = env.rewards[current_agent]
    print("Reward: {}".format(reward))
    #Cumuative reward obtained by agent
    cum_reward = env._cumulative_rewards[current_agent]
    print("Cumulative reward: {}".format(cum_reward))
    #New agent position
    current_agent_position = env.agents_locations[env.agent_selection]
    print("New agent position: {}".format(current_agent_position))
    print("#############################################")
"""

#Test with ray rllib
ray.init()

#Convert pettingzoo environment to ray comaptible environment
env_creator = lambda config: PettingZooGraphEnv()
register_env("pettingzoo_graph_env", lambda config: PettingZooEnv(env_creator(config)))

config = ppo.PPOConfig()

algo = config.build(env="pettingzoo_graph_env")
"""
"""
for i in range(10):
    print(algo.train())


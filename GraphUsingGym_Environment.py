import gym
import graph
import random
import numpy as np

from gym.utils.env_checker import check_env
from gym.wrappers import FlattenObservation

########################
#Class definition

class GymGraphEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}
    
    def __init__(self, render_mode=None, size=10):
        
        self._setup_new_episode()
                
        #Action space is the number of possible moves from a node: "Stay", "Up", "Down", "Left", "Right"
        self.action_space = gym.spaces.Discrete(5)
        
        #Check that the render is either None or a valid render mode from the metadata
        #otherwise assign it to None for safety
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            #If the render mode is not valid assign it to None for safety
            render_mode = None
        self.render_mode = render_mode
        
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
    def _get_obs(self):
        #Return the updated observation space after the agent has moved
        obs = {}
        #Update agent node position using x,y coordinates in the graph and convert it from tuple to box space
        obs["current_agent_node"] = np.array([self.current_agent_node[0], self.current_agent_node[1]])
        #Use old targets nodes positions since targets are not moving
        for i in range(len(self.target_nodes)):
            obs["target_node_" + str(i) + "_position"] = np.array([self.target_nodes[i][0], self.target_nodes[i][1]])
        #Convert the dictionary into a gym dictionary space
        return obs
        
            
    def _get_info(self):
        #return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
        pass
    
    def reset(self, seed=None, options=None):
        
        self._setup_new_episode()
            
        observation = self._get_obs()
        #info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation
    
    def step(self, action):
        
        #action = actions[0]
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
        
        node_color = self.graph.G.nodes[self.current_agent_node]['color']
        if node_color == 'r':
            self.target_nodes_idleness[self.current_agent_node] = 0
        
        #Continue simulation for X steps and then reset the environment
        #Self.steps defines an episode length
        if self.steps >= 10000:
            done = True
        else:
            done = False
            self.steps += 1
        #Otherwise use a different stopping condition for the boolean done variable
        
        """
        #reward is the total number of steps passed since each target node (nodes with red color) has been visited
        #reward is negative to encourage the agent to visit all target nodes and
        #not just the ones that have been visited the longest
        # Calculate the reward for each target node based on its idleness
        reward_part1 = -sum(min(10, idleness) for idleness in self.target_nodes_idleness.values())
        # Penalty for not visiting any of the target nodes
        reward_part2 = -len([node for node in self.target_nodes if self.target_nodes_idleness[node] == 0])
    
        reward = max(reward_part1 + reward_part2, -10)
        """
        
        #Simple reward function for testing
        #0 if agent is not on a target node, 1 if it is but it is not the same node as the previous step
        reward = 0 if self.current_agent_node not in self.target_nodes else 1 if self.current_agent_node != self.previous_agent_node else 0
        self.previous_agent_node = self.current_agent_node
        
        # Increase idleness of all target nodes by 1
        self.target_nodes_idleness = {node: idleness + 1 for node, idleness in self.target_nodes_idleness.items()}
            
        #self.target_nodes_idleness = {node: self.target_nodes_idleness[node] + 1 for node in self.target_nodes_idleness}
                
        return self._get_obs(), reward, done, {}
                    
    def _setup_new_episode(self):
        
        #Per ogni episodio servono:
        #Reset steps
        #Reset total idleness
        #1) Un nuovo grafo
        #2) Un nuovo nodo di partenza
        #3) Una nuova lista di nodi target
        #4) Observation space aggiornato
        
        
        #Reset steps
        self.steps = 0
        #Reset total idleness
        self.total_idleness = 0
        
        #Generate new graph
        self.graph = graph.Graph(10, 10)
        
        #Select new random node as the agent's starting location
        self.current_agent_node = random.choice(list(self.graph.G.nodes))
        
        self.previous_agent_node = self.current_agent_node
        
        #Retrieve new list of target nodes (nodes with red color)
        self.target_nodes = [node for node in self.graph.G.nodes if self.graph.G.nodes[node]['color'] == 'red']
        #Determine the position of each target node and assign it to a dictionary for gym observation space
        self.target_nodes_positions_dict = {node: self.graph.G.nodes[node]['position'] for node in self.target_nodes}
        
        #Set target nodes idleness to 0
        self.target_nodes_idleness = {node: 0 for node in self.target_nodes}
        
        #Create dictionary to convert node to index
        self.node_to_index = {node: index for index, node in enumerate(self.graph.G.nodes)}
        
        #print ("New episode started")
        
        #Adapt graph to gym's observation space format:
        self.observation_space = gym.spaces.Dict()
        #1) Add the position of the agent node to the observation space as x, y coordinates in the graph
        self.observation_space['current_agent_node'] = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.graph.n, self.graph.m]), dtype=np.int32)
        #2) Add the position of each target node to the observation space as x, y coordinates in the graph
        for i, node in enumerate(self.target_nodes):
            dictionary_key = 'target_node_' + str(i) + '_position'
            self.observation_space[dictionary_key] = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.graph.n, self.graph.m]), dtype=np.int32)           
        
        #self.temp_obs_space['target_nodes_positions'] = gym.spaces.Dict({node: gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.graph.n, self.graph.m]), dtype=np.int32) for node in self.target_nodes})

    def render(self, mode='human'):
        #During the drawing of the graph, the agent's current node is highlighted in green
        self.graph.G.nodes[self.current_agent_node]['color'] = 'green'
        self.graph.draw()
        #pass
        
    
    
#########################
#Class testing
"""
env = GymGraphEnv()

env.graph.draw()

print(env.observation_space)

#Using wrapper to convert observation space to gym's format
wrapped_env = gym.wrappers.FlattenObservation(env)
check_env(wrapped_env)
"""
        
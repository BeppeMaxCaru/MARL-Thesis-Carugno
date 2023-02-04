import gym
import graph
import random
import numpy as np

########################
#Class definition

class GymGraphEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}
    
    def __init__(self, render_mode=None, size=10):
        
        self._setup_new_episode()
        
        #Action space is the number of possible moves from a node: "Stay", "Up", "Down", "Left", "Right"
        self.action_space = gym.spaces.Discrete(5)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
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
        
        return {"agent": self.current_node, "targets": self.target_nodes_dict}
            
    def _get_info(self):
        #return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
        pass
    
    def reset(self, seed=None, options=None):
        
        self._setup_new_episode()
            
        observation = self._get_obs()
        #info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation#, info
    
    def step(self, action):
        
        #action = actions[0]
        if action == 0:
            #Stay in the same node
            pass
        elif action == 1:
            #Move to the node above in the graph if edge going to it exists
            #Check if the node above exists and if there is an edge going to it
            if (self.current_node[0] > 0) and (self.graph.G.has_edge(self.current_node, (self.current_node[0]-1, self.current_node[1]))):
                #Move to the node above
                self.current_node = (self.current_node[0]-1, self.current_node[1])
        elif action == 2:
            #Move to the node down in the graph if edge exists
            #Check if the node below exists and if there is an edge going to it
            if (self.current_node[0] < self.graph.n-1) and (self.graph.G.has_edge(self.current_node, (self.current_node[0]+1, self.current_node[1]))):
                #Move to the node below
                self.current_node = (self.current_node[0]+1, self.current_node[1])
        elif action == 3:
            #Move to the node on the left in the graph if edge exists
            #Check if the node to the left exists and if there is an edge going to it
            if (self.current_node[1] > 0) and (self.graph.G.has_edge(self.current_node, (self.current_node[0], self.current_node[1]-1))):
                #Move to the node to the left
                self.current_node = (self.current_node[0], self.current_node[1]-1)
        elif action == 4:
            #Move to the node on the right in the graph if edge exists
            #Check if the node to the right exists and if there is an edge going to it
            if (self.current_node[1] < self.graph.m-1) and (self.graph.G.has_edge(self.current_node, (self.current_node[0], self.current_node[1]+1))):
                #Move to the node to the right
                self.current_node = (self.current_node[0], self.current_node[1]+1)
        else:
            #Invalid action
            raise ValueError("Invalid action: {}".format(action))
        
        node_color = self.graph.G.nodes[self.current_node]['color']
        if node_color == 'r':
            self.target_nodes[self.current_node] = 0
        
        #Continue simulation for 100 steps
        if self.steps >= 100:
            done = True
        else:
            done = False
            self.steps += 1
        
        #reward is the total number of steps passed since each target node (nodes with red color) has been visited
        #reward is negative to encourage the agent to visit all target nodes and
        #not just the ones that have been visited the longest
        reward = - self.total_idleness() - len([node for node in self.target_nodes if self.target_nodes_idleness[node] == 0])
            
        #Update the number of steps passed since each target node (nodes with red color) has been visited
        #Increase idleness of all target nodes by 1
        self.target_nodes_idleness = {node: self.target_nodes_idleness[node] + 1 for node in self.target_nodes_idleness}
        
        return {
            #observation is a list of the number of steps passed since each target node (nodes with red color) has been visited
            "observation": [self.target_nodes[node] for node in self.target_nodes],
            "reward": reward,
            "done": done
        }
        
    def total_idleness(self):
        return sum(self.target_nodes.values())
    
    
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
        #Retrieve new list of target nodes (nodes with red color)
        self.target_nodes = [node for node in self.graph.G.nodes if self.graph.G.nodes[node]['color'] == 'red']
        #Determine the position of each target node and assign it to a dictionary for gym observation space
        self.target_nodes_positions_dict = {node: self.graph.G.nodes[node]['position'] for node in self.target_nodes}
        
        #Adapt graph to gym's observation space format:
        self.observation_space = gym.spaces.Dict()
        #1) Convert agent's node position to a key
        self.observation_space['agent_node'] = gym.spaces.Discrete(len(self.graph.G.nodes))
        #2) Add all target nodes positions to the observation space
        self.observation_space['targets_nodes'] = gym.spaces.Dict({node: gym.spaces.Discrete(len(self.graph.G.nodes)) for node in self.target_nodes_positions_dict})

#########################
#Class testing
env = GymGraphEnv()

env.graph.draw()

print(env.current_agent_node)
print(env.observation_space)

        
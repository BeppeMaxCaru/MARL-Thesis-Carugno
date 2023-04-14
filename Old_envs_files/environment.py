#imported libraries
import pettingzoo
import gymnasium
import random

#manually defined modules
import graph

class myEnvironment(pettingzoo.AECEnv):        
    metadata = {"render_modes": ["human"], "name": "Patrolling Environment"}
    
    #General execution flow:
    #0. __init__ to initialize the environment
    #1. _setup to initialize the environment's variables and settings 
    #Then loop between reset and step
    #2. _reset to reset the environment to initial state to start a new episode
    #3. _step until the episode is done and then _reset again for a new episode
    
    #Environment initialization
    def __init__(self, num_patrollers=1, num_attackers=1, render_mode=None):
        """
        Executes the environment's initialization
        """
        #Simulation steps
        self.steps = 0

        #Generate new graph and select a new starting position
        self.graph = graph.Graph(10, 10)
        self.current_node = random.choice(list(self.graph.G.nodes()))
        #Make a dictionary that counts how many steps have passed for only red nodes since each one has been visited
        self.target_nodes = {node: 0 for node in self.graph.G.nodes() if self.graph.G.nodes[node]['color'] == 'red'}
        
        #Agents setup
        self.num_patrollers = num_patrollers
        self.num_attackers = num_attackers

        self.possible_agents = []
        
        #Add patroller agents
        for i in range(num_patrollers):
            agent_ID = 'patroller_' + str(i)
            self.possible_agents.append(agent_ID)
        
        #Add attacker agents
        for i in range(num_attackers):
            agent_ID = 'attacker_' + str(i)
            self.possible_agents.append(agent_ID)

        #Agent types
        self.AGENT_TYPE_PATROLLER = "patroller"
        self.AGENT_TYPE_ATTACKER = "attacker"
        self.agent_types = [self.AGENT_TYPE_PATROLLER, self.AGENT_TYPE_ATTACKER]
        
        #Action space is a dictionary of agent names to action spaces
        #Possible actions are 0: stay, 1: up, 2: down, 3: left, 4: right
        self.action_space = {}
        for agent_name in self.possible_agents:
            agent_type = agent_name.split("_")[0]
            if agent_type == self.AGENT_TYPE_PATROLLER:
                self.action_space[agent_name] = gymnasium.spaces.Discrete(5)
            elif agent_type == self.AGENT_TYPE_ATTACKER:
                self.action_space[agent_name] = gymnasium.spaces.Discrete(5)
                
        #Observation space is a dictionary of agent names to observation spaces
        #Observation space is composed of:
        #The current node of the graph on which the agent is located
        #The position of each target node
        #The idleness of each target node
        self.observation_space = {}
        for agent_name in self.possible_agents:
            agent_type = agent_name.split("_")[0]
            if agent_type == self.AGENT_TYPE_PATROLLER:
                #observation space is composed of: 
                self.observation_space[agent_name] = gymnasium.spaces.Dict({
                    'current_node': gymnasium.spaces.Discrete(len(self.graph.G.nodes)),
                    'targets_pos': gymnasium.spaces.Discrete(len(self.graph.G.nodes)),
                    'targets_idleness': gymnasium.spaces.Box(low=0, high=100, shape=(len(self.target_nodes),), dtype=int)
                    })            
            elif agent_type == self.AGENT_TYPE_ATTACKER:
                self.observation_space[agent_name] = gymnasium.spaces.Dict({
                    'current_node': gymnasium.spaces.Discrete(len(self.graph.G.nodes)),
                    'targets_pos': gymnasium.spaces.Discrete(len(self.graph.G.nodes)),
                    'targets_idleness': gymnasium.spaces.Box(low=0, high=100, shape=(len(self.target_nodes),), dtype=int)
                    })
        
        #Render simulation in real time
        self.render_mode = render_mode
               
    #Reset the environment              
    def reset(self):
        """
        Resets the environment and returns the initial observation, reward and done status.
        :return: A dictionary containing the observation, reward and done status of the environment.
        """
        #Generate new graph and select a new starting position for the agent
        self.graph = graph.Graph(10, 10)
        self.current_node = random.choice(list(self.graph.G.nodes()))
        #Make a dictionary that counts how many steps have passed for only red nodes since each one has been visited
        self.target_nodes = {node: 0 for node in self.graph.G.nodes() if self.graph.G.nodes[node]['color'] == 'red'}
        
        #Update observation space to match the new graph
        self.observation_space = {}
        for agent_name in self.possible_agents:
            agent_type = agent_name.split("_")[0]
            if agent_type == self.AGENT_TYPE_PATROLLER:
                #observation space is composed of: 
                self.observation_space[agent_name] = gymnasium.spaces.Dict({
                    'current_node': gymnasium.spaces.Discrete(len(self.graph.nodes)),
                    'targets_pos': gymnasium.spaces.Discrete(len(self.graph.nodes)),
                    'targets_idleness': gymnasium.spaces.Box(low=0, high=100, shape=(len(self.target_nodes),), dtype=int)
                    })            
            elif agent_type == self.AGENT_TYPE_ATTACKER:
                self.observation_space[agent_name] = gymnasium.spaces.Dict({
                    'current_node': gymnasium.spaces.Discrete(len(self.graph.nodes)),
                    'targets_pos': gymnasium.spaces.Discrete(len(self.graph.nodes)),
                    'targets_idleness': gymnasium.spaces.Box(low=0, high=100, shape=(len(self.target_nodes),), dtype=int)
                    })
                
        #Return values
        return {
            #observation is a list of the number of steps passed since each target node (nodes with red color) has been visited
            "observation": [self.target_nodes[node] for node in self.target_nodes],
            #reward is the total number of steps passed since each target node (nodes with red color) has been visited
            #reward is negative to encourage the agent to visit all target nodes and
            #not just the ones that have been visited the longest
            "reward": - self.total_idleness() - len([node for node in self.target_nodes if self.target_nodes[node] == 0]),
            "done": False
        }
    
    def total_idleness(self):
        return sum(self.target_nodes.values())
                
    def step(self, actions):
        action = actions[0]
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
        reward = - self.total_idleness() - len([node for node in self.target_nodes if self.target_nodes[node] == 0])
            
        #Update the number of steps passed since each target node (nodes with red color) has been visited
        #Increase idleness of all target nodes by 1
        self.target_nodes = {node: self.target_nodes[node] + 1 for node in self.target_nodes}
        
        return {
            #observation is a list of the number of steps passed since each target node (nodes with red color) has been visited
            "observation": [self.target_nodes[node] for node in self.target_nodes],
            "reward": reward,
            "done": done
        }
        
    def render(self, mode='human'):
        #Render the graph
        self.graph.render()
        
        #Render the current node
        print("Current node: {}".format(self.current_node))
        
        #Render the number of steps passed since each target node (nodes with red color) has been visited
        print("Idleness of target nodes: {}".format(self.target_nodes))
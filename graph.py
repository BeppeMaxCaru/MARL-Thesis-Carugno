import random
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.G = nx.grid_2d_graph(n, m)
        self.pos = {}
        for i in range(n):
            for j in range(m):
                self.pos[i, j] = (i, j)
        #Assign the positions the nodes in the graph
        for node in self.G.nodes():
            self.G.nodes[node]['position'] = self.pos[node]
        #Randomize graph
        self.remove_random_nodes_and_edges(0.2)
        #Randomize target nodes
        self.colour_targets_nodes(0.2)
        #self.draw()
                
    def remove_random_nodes_and_edges(self, p=0.5):
        nodes_to_remove = random.sample(list(self.G.nodes()), int(p * self.G.number_of_nodes()))
        for node in nodes_to_remove:
            G_copy = self.G.copy()
            G_copy.remove_node(node)
            if nx.is_connected(G_copy):
                self.G.remove_node(node)
        edges_to_remove = random.sample(list(self.G.edges()), int(p * self.G.number_of_edges()))
        for edge in edges_to_remove:
            G_copy = self.G.copy()
            G_copy.remove_edge(*edge)
            if nx.is_connected(G_copy):
                self.G.remove_edge(*edge)
        return self.G
    
    def colour_targets_nodes(self, coloring_pct=0.2):
        # Get a list of the remaining nodes after removing nodes
        remaining_nodes = list(self.G.nodes())
        # Get a random sample of nodes to color
        color_nodes = random.sample(remaining_nodes, int(len(remaining_nodes) * coloring_pct))
        # Create a list of node-color pairs for nodes
        for node in remaining_nodes:
            if node in color_nodes:
                # Assign the color red to the node
                self.G.nodes[node]['color'] = 'red'
            else:
                # Assign the color black to the node
                self.G.nodes[node]['color'] = 'black'
        nx.draw_networkx(self.G, self.pos, node_size=100, node_color=[self.G.nodes[node]['color'] for node in remaining_nodes], edge_color='k', with_labels=False)
        return self.G
    
    def draw(self):
        #nx.draw_networkx(self.G, self.pos, node_size=100, node_color='k', edge_color='k', with_labels=False)
        plt.show()
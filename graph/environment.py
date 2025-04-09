import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from graph.draw import visualize_network
import pdb

class Environment:
    def __init__(self, num_nodes=6, connection_prob=0.3, max_tries=100):
        if num_nodes < 2:
            raise ValueError("The number of nodes must be at least 2")
        self.num_nodes = num_nodes
        self.connection_prob = connection_prob
        self.max_tries = max_tries
        self.directed_graph = nx.DiGraph()

        self._create_random_graph()
        self._assign_node_attributes()
        self._assign_edge_attributes()
    
    def _create_random_graph(self):
        """
        Try generating the random bidirectional network until success or max_tries is reached
        """
        for attempt in range(self.max_tries):
            random_graph = nx.gnp_random_graph(self.num_nodes, self.connection_prob)
            # no disjoint/isolated node is allowed
            if not nx.is_connected(random_graph):
                continue

            # build directional edges
            self.directed_graph.clear()
            self.directed_graph.add_nodes_from(random_graph.nodes())
            for u, v in random_graph.edges():
                self.directed_graph.add_edge(u, v)
                self.directed_graph.add_edge(v, u)
            break
        else:
            raise RuntimeError(
                f"Failed to generate a valid random graph after {self.max_tries} attempts"
            )
    
    def _assign_node_attributes(self):
        """
        - node_type: food, nest, distractor
        - position: (x, y) coordinate
        """
        for attempt in range(self.max_tries):
            nodes = list(self.directed_graph.nodes())

            nest = random.choice(nodes)
            if self.num_nodes > 2:
                # exclude any node that is directly connected to or from the nest node
                food_candidates = [
                    n for n in nodes 
                    if n != nest and not (self.directed_graph.has_edge(nest, n) or self.directed_graph.has_edge(n, nest))
                ]
            else:
                # for a 2-node graph, the only candidate is the node that is not the nest
                food_candidates = [n for n in nodes if n != nest]
            
            if not food_candidates:
                print(f"Attempt {attempt+1} of regenerating random graph")
                self._create_random_graph()
                continue

            food = random.choice(food_candidates)
            
            for node in nodes:
                if node == nest:
                    self.directed_graph.nodes[node]['node_type'] = 'nest'
                elif node == food:
                    self.directed_graph.nodes[node]['node_type'] = 'food'
                else:
                    self.directed_graph.nodes[node]['node_type'] = 'distractor'
            
            # assign random positions to calculate direction and distance
            for node in nodes:
                x = random.uniform(0, 100)
                y = random.uniform(0, 100)
                self.directed_graph.nodes[node]['position'] = (x, y)
            
            if nx.is_connected(self.directed_graph.to_undirected()):
                return
            else:
                print(f"Attempt {attempt+1}: Graph is disconnected after assignment. Regenerating random graph.")
                self._create_random_graph()
        
        raise RuntimeError("Failed to assign node attributes after maximum attempts")


    def _assign_edge_attributes(self):
        """
        For each directed edge:
            - distance: Euclidean distance between node positions
            - direction: One of 8 quadrants (N, NE, E, SE, S, SW, W, NW)
        """
        
        for (u, v) in self.directed_graph.edges():
            pos_u = self.directed_graph.nodes[u].get('position')
            pos_v = self.directed_graph.nodes[v].get('position')
            distance = self.calculate_distance(pos_u, pos_v)
            direction = self.calculate_direction(pos_u, pos_v)
            self.directed_graph.edges[u, v]['distance'] = distance
            self.directed_graph.edges[u, v]['direction'] = direction
    
    def calculate_distance(self, pos1, pos2):
        """Calculates Euclidean distance between two points"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.hypot(dx, dy)
    
    def calculate_direction(self, pos1, pos2):
        """
        Computes a direction into one of the 8 quadrants
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360

        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int((angle + 22.5) // 45) % 8
        return directions[idx]

if __name__ == '__main__':
    env = Environment(num_nodes=2, connection_prob=0.5)

    nodes = list(env.directed_graph.nodes(data=True))
    node_types = {node: data['node_type'] for node, data in nodes}

    connections = list(env.directed_graph.edges(data=True))
    for (u, v, attr) in connections:
        print(f"Edge {u} ({node_types[u]}) -> {v} ({node_types[v]}): distance={attr['distance']:.2f}, direction={attr['direction']}")
    
    visualize_network(env)

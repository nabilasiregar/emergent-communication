import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from draw import visualize_network
import pdb

class Environment:
    def __init__(self, num_nodes=6, connection_prob=0.3, max_tries=100):
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
            food = random.choice([n for n in nodes if n != nest])
            
            # pick nest and food nodes randomly, the remaining nodes are distractors
            for node in nodes:
                if node == nest:
                    self.directed_graph.nodes[node]['node_type'] = 'nest'
                elif node == food:
                    self.directed_graph.nodes[node]['node_type'] = 'food'
                else:
                    self.directed_graph.nodes[node]['node_type'] = 'distractor'
            
            # assign random positions within a 100x100 area (to calculate direction and distance)
            for node in nodes:
                x = random.uniform(0, 100)
                y = random.uniform(0, 100)
                self.directed_graph.nodes[node]['position'] = (x, y)
            
            # if nest and food are directly connected, remove that connection
            removed_edges = []
            if self.directed_graph.has_edge(nest, food):
                self.directed_graph.remove_edge(nest, food)
                removed_edges.append((nest, food))
            if self.directed_graph.has_edge(food, nest):
                self.directed_graph.remove_edge(food, nest)
                removed_edges.append((food, nest))
            
            # if connection is removed, check connectivity
            if removed_edges:
                if nx.is_connected(self.directed_graph.to_undirected()):
                    print(f"Node attributes assigned (attempt {attempt+1}). "
                          f"Removed direct edge(s) between nest {nest} and food {food}.")
                    return
                else:
                    # regenerate a new random graph
                    for u, v in removed_edges:
                        self.directed_graph.add_edge(u, v)
                    print(f"Removing direct edge between nest {nest} and food {food} broke connectivity"
                          f"(attempt {attempt+1}). Regenerating a new random graph.")
                    self._create_random_graph()
                    continue
            else:
                print(f"Node attributes assigned (attempt {attempt+1})"
                      f"No direct edge between nest {nest} and food {food}")
                return
        
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
    env = Environment(num_nodes=4, connection_prob=0.5)

    nodes = list(env.directed_graph.nodes(data=True))
    node_types = {node: data['node_type'] for node, data in nodes}

    connections = list(env.directed_graph.edges(data=True))
    for (u, v, attr) in connections:
        print(f"Edge {u} ({node_types[u]}) -> {v} ({node_types[v]}): distance={attr['distance']:.2f}, direction={attr['direction']}")
    
    visualize_network(env)

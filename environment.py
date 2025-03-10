import math
import random
import networkx as nx
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, num_nodes=6, connection_prob=0.3, max_tries=100):
        self.num_nodes = num_nodes
        self.connection_prob = connection_prob
        self.max_tries = max_tries
        self.graph = nx.DiGraph()
        self.nest = None
        self.food = None

        self._setup_environment()

    def _setup_environment(self):
        """
        Try generating the random bidirectional network until success or max_tries is reached.
        """
        for attempt in range(self.max_tries):
            # generate a connected random graph
            random_graph = nx.erdos_renyi_graph(self.num_nodes, self.connection_prob)
            if not nx.is_connected(random_graph):
                continue  

            # pick nest and food nodes randomly
            all_nodes = list(random_graph.nodes())
            random.shuffle(all_nodes)
            nest_node, food_node = random.sample(all_nodes, 2)

            # if nest and food are directly connected, remove that edge
            if random_graph.has_edge(nest_node, food_node):
                random_graph.remove_edge(nest_node, food_node)

                # if removing the edge broke connectivity, revert and regenerate the graph
                if not nx.is_connected(random_graph):
                    continue

            self.nest = nest_node
            self.food = food_node

            # build the bidirections
            self.graph.clear()
            for node in random_graph.nodes():
                # assign node type and random positions to calculate distance
                x = random.uniform(-100, 100)
                y = random.uniform(-100, 100)
                self.graph.add_node(node, type='Distractor', pos=(x, y))

            for u, v in random_graph.edges():
                self.graph.add_edge(u, v)
                self.graph.add_edge(v, u)

            self.graph.nodes[self.nest]['type'] = 'Nest'
            self.graph.nodes[self.food]['type'] = 'Food'

            # there must be a directed path from nest to food, otherwise regenerate the graph
            if nx.has_path(self.graph, self.nest, self.food):
                self._assign_edge_attributes()
                return
            else:
                continue

        raise RuntimeError(
            f"Failed to generate a valid graph after {self.max_tries} attempts"
        )

    def _assign_edge_attributes(self):
        """
        For each directed edge:
          - distance: Euclidean distance between node positions
          - direction: One of 8 quadrants (N, NE, E, SE, S, SW, W, NW)
        """
        for (u, v) in self.graph.edges():
            pos_u = self.graph.nodes[u]['pos']
            pos_v = self.graph.nodes[v]['pos']
            distance = self.calculate_distance(pos_u, pos_v)
            direction = self.calculate_direction(pos_u, pos_v)
            self.graph[u][v]['distance'] = distance
            self.graph[u][v]['direction'] = direction
    
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

def visualize_environment(env):
    pos = nx.get_node_attributes(env.graph, 'pos')
    types = nx.get_node_attributes(env.graph, 'type')

    color_map = {'Nest': 'gold', 'Food': 'red', 'Distractor': 'skyblue'}
    node_colors = [color_map.get(types[n], 'gray') for n in env.graph.nodes()]
    labels = {n: f"{types[n]}" for n in env.graph.nodes()}

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(env.graph, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(env.graph, pos, labels, font_size=10)
    
    for (u, v, d) in env.graph.edges(data=True):
        nx.draw_networkx_edges(env.graph, pos, edgelist=[(u, v)], arrowstyle='-|>', arrowsize=15,
                            edge_color='gray', connectionstyle='arc3,rad=0.1')
        edge_label = { (u, v): f"{d['distance']:.1f}m ({d['direction']})" }
        nx.draw_networkx_edge_labels(env.graph, pos, edge_labels=edge_label, font_color='black', label_pos=0.4, font_size=8, connectionstyle='arc3,rad=0.1')
        
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    env = Environment(num_nodes=4, connection_prob=0.5)
    print(f"Nest node ID: {env.nest}, Food node ID: {env.food}")

    connections = list(env.graph.edges(data=True))
    for (u, v, attr) in connections:
        print(f"Edge {u}->{v}: distance={attr['distance']:.2f}, direction={attr['direction']}")
    
    visualize_environment(env)
    